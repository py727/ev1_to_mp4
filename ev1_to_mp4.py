#!/usr/bin/env python3
"""Attempt to restore a Baijiayun EV1 file and convert it to MP4 with FFmpeg.

Based on the visible BPlayer SDK behavior: EV1 is mapped to FLV and the first
100 bytes are bitwise-inverted in the loader callback. This is an experimental
converter; it does not implement undocumented authentication/DRM/key workflows.

Examples:
  python ev1_to_mp4.py lesson.ev1
  python ev1_to_mp4.py lesson.ev1 -o lesson.mp4 --play
  python ev1_to_mp4.py lesson.ev1 --mode transcode --keep-flv
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MASK_BYTES = 100
PLAIN_FLV_MAGIC = b"FLV"
MASKED_FLV_MAGIC = bytes((b ^ 0xFF) for b in PLAIN_FLV_MAGIC)


class ConversionError(RuntimeError):
    pass


def print_cmd(cmd: list[str]) -> None:
    if os.name == "nt":
        rendered = subprocess.list2cmdline(cmd)
    else:
        rendered = shlex.join(cmd)
    print(f"\n$ {rendered}")


def run_command(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print_cmd(cmd)
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )


def find_media_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise ConversionError(
            "没有找到 ffmpeg。请先安装 FFmpeg，并将 ffmpeg 加入 PATH。"
        )
    if not ffprobe:
        raise ConversionError(
            "没有找到 ffprobe。通常它随 FFmpeg 一起提供，请检查 PATH。"
        )
    return ffmpeg, ffprobe


def write_candidate(source: Path, destination: Path, *, invert_prefix: bool) -> None:
    """Copy a potentially large file without loading it entirely into RAM."""
    with source.open("rb") as src, destination.open("wb") as dst:
        prefix = src.read(MASK_BYTES)
        if invert_prefix:
            prefix = bytes(value ^ 0xFF for value in prefix)
        dst.write(prefix)
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def probe_file(ffprobe: str, path: Path, *, force_flv: bool = False) -> dict[str, Any] | None:
    cmd = [ffprobe, "-v", "error"]
    if force_flv:
        cmd += ["-f", "flv"]
    cmd += [
        "-show_entries",
        "format=format_name,duration,size:stream=index,codec_type,codec_name,width,height,sample_rate,channels",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        print(f"[probe] 无法识别 {path.name}: {detail[-500:]}")
        return None
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"[probe] ffprobe 返回的内容不是有效 JSON：{path.name}")
        return None
    streams = info.get("streams") or []
    if not streams:
        print(f"[probe] 未发现媒体流：{path.name}")
        return None
    return info


def print_media_info(info: dict[str, Any], label: str) -> None:
    fmt = info.get("format") or {}
    print(f"\n[{label}] format={fmt.get('format_name', 'unknown')}, duration={fmt.get('duration', 'unknown')}s")
    for stream in info.get("streams", []):
        kind = stream.get("codec_type", "unknown")
        codec = stream.get("codec_name", "unknown")
        if kind == "video":
            print(f"  stream #{stream.get('index')}: video, codec={codec}, {stream.get('width')}x{stream.get('height')}")
        elif kind == "audio":
            print(f"  stream #{stream.get('index')}: audio, codec={codec}, sample_rate={stream.get('sample_rate')}, channels={stream.get('channels')}")
        else:
            print(f"  stream #{stream.get('index')}: {kind}, codec={codec}")


def has_video(info: dict[str, Any]) -> bool:
    return any(s.get("codec_type") == "video" for s in info.get("streams", []))


def tail_text(text: str | None, max_lines: int = 20) -> str:
    if not text:
        return "(没有错误详情)"
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def try_ffmpeg_conversion(
    ffmpeg: str,
    ffprobe: str,
    flv_path: Path,
    output_path: Path,
    *,
    transcode: bool,
) -> bool:
    temp_output = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
    try:
        temp_output.unlink(missing_ok=True)
        cmd = [ffmpeg, "-hide_banner", "-y", "-loglevel", "warning", "-fflags", "+genpts", "-f", "flv", "-i", str(flv_path),
               "-map", "0:v:0", "-map", "0:a?", "-movflags", "+faststart"]
        if transcode:
            # Encoding fallback for codecs/packet layouts that cannot be copied into MP4.
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k"]
        else:
            cmd += ["-c", "copy"]
        cmd += [str(temp_output)]
        result = run_command(cmd, capture=True)
        if result.returncode != 0 or not temp_output.exists() or temp_output.stat().st_size == 0:
            print(f"[ffmpeg] {'转码' if transcode else '直接封装'}失败：\n{tail_text(result.stderr)}")
            return False
        output_info = probe_file(ffprobe, temp_output)
        if not output_info or not has_video(output_info):
            print("[check] 输出文件未通过验证（未发现有效视频流）。")
            return False
        temp_output.replace(output_path)
        print_media_info(output_info, "MP4 输出验证通过")
        return True
    finally:
        temp_output.unlink(missing_ok=True)


def open_for_playback(path: Path, ffmpeg: str) -> None:
    ffplay = shutil.which("ffplay")
    if ffplay:
        print("\n正在使用 ffplay 打开转换后的 MP4；关闭播放器窗口后脚本继续。")
        subprocess.run([ffplay, "-autoexit", str(path)], check=False)
        return

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            opener = shutil.which("xdg-open")
            if not opener:
                raise ConversionError("未找到 ffplay 或 xdg-open，无法自动打开播放器。")
            subprocess.Popen([opener, str(path)])
        print(f"已请求系统默认播放器打开：{path}")
    except OSError as exc:
        raise ConversionError(f"自动播放失败：{exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="尝试还原百家云 EV1 数据头，并通过 FFmpeg 转换为 MP4。"
    )
    parser.add_argument("input", type=Path, help="输入 .ev1 文件")
    parser.add_argument("-o", "--output", type=Path, help="输出 MP4 路径（默认与输入同名）")
    parser.add_argument(
        "--mode", choices=("auto", "remux", "transcode"), default="auto",
        help="auto=先无损封装，失败后转码；remux=只尝试无损封装；transcode=直接转码（默认 auto）",
    )
    parser.add_argument("--keep-flv", action="store_true", help="保留还原后的 .flv 中间文件，便于排查")
    parser.add_argument("--play", action="store_true", help="转换成功后尝试打开 MP4 播放")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source: Path = args.input.expanduser().resolve()
    output: Path = (args.output.expanduser().resolve() if args.output else source.with_suffix(".mp4"))

    if not source.is_file():
        print(f"错误：输入文件不存在：{source}", file=sys.stderr)
        return 2
    if source == output:
        print("错误：输出路径不能与输入文件相同。", file=sys.stderr)
        return 2
    if source.stat().st_size < 9:
        print("错误：文件太小，不像是有效的视频文件。", file=sys.stderr)
        return 2

    try:
        ffmpeg, ffprobe = find_media_tools()
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"输入：{source} ({source.stat().st_size:,} bytes)")
        print(f"输出：{output}")
        print("策略：先尝试按 BPlayer 源码还原 EV1 前 100 字节，再交给 FFmpeg 按 FLV 解析。")

        with tempfile.TemporaryDirectory(prefix="ev1_convert_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            with source.open("rb") as f:
                first3 = f.read(3)

            candidates: list[tuple[str, Path, bool]] = []
            if first3 == PLAIN_FLV_MAGIC:
                plain = temp_dir / "input_plain.flv"
                write_candidate(source, plain, invert_prefix=False)
                candidates.append(("原文件已包含 FLV 标识（不修改）", plain, False))
            else:
                restored = temp_dir / "input_restored.flv"
                write_candidate(source, restored, invert_prefix=True)
                if first3 == MASKED_FLV_MAGIC:
                    print("检测到与 FLV 标识按位取反相符的 EV1 前缀。")
                else:
                    print("未识别到标准 FLV 或已知掩码前缀；仍尝试还原前 100 字节。")
                candidates.append(("EV1 前 100 字节取反", restored, True))
                original = temp_dir / "input_original.flv"
                write_candidate(source, original, invert_prefix=False)
                candidates.append(("备用：原始字节不修改", original, False))

            selected_flv: Path | None = None
            selected_label = ""
            for label, candidate_path, _was_inverted in candidates:
                print(f"\n正在检测候选：{label}")
                info = probe_file(ffprobe, candidate_path, force_flv=True)
                if info and has_video(info):
                    selected_flv = candidate_path
                    selected_label = label
                    print_media_info(info, "FLV 输入检测通过")
                    break
                print("该候选未通过视频流检测，继续尝试下一种方式。")

            if selected_flv is None:
                raise ConversionError(
                    "FFprobe 无法从还原后或原始数据中识别有效视频流。该文件可能不是这种 EV1 变体，可能还需要厂商专用解密/授权，或文件已损坏。"
                )

            success = False
            if args.mode in ("auto", "remux"):
                print("\n尝试直接封装（不重新编码，-c copy）……")
                success = try_ffmpeg_conversion(ffmpeg, ffprobe, selected_flv, output, transcode=False)
                if not success and args.mode == "remux":
                    raise ConversionError("无损封装失败；你可以改用 --mode auto 或 --mode transcode。")

            if not success and args.mode in ("auto", "transcode"):
                print("\n尝试转码为 H.264 + AAC……")
                success = try_ffmpeg_conversion(ffmpeg, ffprobe, selected_flv, output, transcode=True)

            if not success:
                raise ConversionError(
                    "FFmpeg 未能生成通过验证的 MP4。请查看上方错误；如果 FFprobe 已识别输入流，可尝试更新 FFmpeg 或使用 --mode transcode。"
                )

            if args.keep_flv:
                keep_path = output.with_suffix(".recovered.flv")
                shutil.copy2(selected_flv, keep_path)
                print(f"已保留中间文件（{selected_label}）：{keep_path}")

        print(f"\n转换成功：{output} ({output.stat().st_size:,} bytes)")
        if args.play:
            open_for_playback(output, ffmpeg)
        return 0

    except ConversionError as exc:
        print(f"\n转换失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
