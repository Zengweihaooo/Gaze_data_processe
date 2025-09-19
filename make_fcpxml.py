# make_fcpxml.py
import argparse, os, sys, re, glob, xml.etree.ElementTree as ET
from fractions import Fraction
from urllib.parse import quote

def file_url(path):
    p = os.path.abspath(path).replace("\\", "/")
    return "file:///" + quote(p) if os.name == "nt" else "file://" + quote(p)

def inv(frac: Fraction) -> Fraction:
    return Fraction(frac.denominator, frac.numerator)

def frac_to_s(frac: Fraction) -> str:
    return f"{frac.numerator}/{frac.denominator}s"

def natural_key(name: str):
    # 拆成“非数字/数字”段，数字转为int，实现 P4_2 < P4_10
    parts = re.split(r'(\d+)', name)
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out

def main():
    ap = argparse.ArgumentParser(description="Make FCPXML with fixed frame offsets.")
    ap.add_argument("folder", help="Folder containing media files")
    ap.add_argument("--fps", type=float, default=60.0, help="Sequence frame rate (e.g., 60, 30, 59.94)")
    ap.add_argument("--step", type=int, default=20000, help="Frame step between clips (e.g., 20000)")
    ap.add_argument("--glob", default="*.mp4", help="Glob pattern, e.g. *.mp4 or P4_*.mp4")
    ap.add_argument("--out", default="timeline.fcpxml", help="Output FCPXML filename")
    args = ap.parse_args()

    # 收集文件并按自然排序
    pattern = os.path.join(args.folder, args.glob)
    files = sorted(glob.glob(pattern), key=lambda p: natural_key(os.path.basename(p)))
    if not files:
        print("No files matched. Try adjusting --glob (e.g., *.mp4 or P4_*.mp4).", file=sys.stderr)
        sys.exit(1)

    # FCPXML 根与资源
    ET.register_namespace('', "http://www.apple.com/fcpxml")
    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")

    # 计算 frameDuration（尽量用分数表达，兼容 59.94/29.97）
    fps_frac = Fraction(str(args.fps)).limit_denominator(100000)
    frame_dur = inv(fps_frac)  # = 1/fps
    fmt = ET.SubElement(resources, "format", id="r1",
                        frameDuration=frac_to_s(frame_dur),
                        width="1920", height="1080", colorSpace="1-1-1 (Rec. 709)")

    # 资产
    asset_ids = []
    for i, path in enumerate(files, 1):
        aid = f"asset{i}"
        asset_ids.append(aid)
        ET.SubElement(resources, "asset",
                      id=aid, name=os.path.basename(path),
                      src=file_url(path), start="0s",
                      hasAudio="1", hasVideo="1", format="r1")

    # 序列与时间线
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="AutoPlaced")
    project = ET.SubElement(event, "project", name=f"Auto_{args.step}f")
    sequence = ET.SubElement(project, "sequence", format="r1")
    spine = ET.SubElement(sequence, "spine")

    # 放置：第 n 个片段 → 偏移 n*step 帧
    for idx, aid in enumerate(asset_ids):
        start_frames = idx * args.step  # idx 从 0 开始时为 0、20000、40000…
        start_seconds = Fraction(start_frames, 1) * frame_dur  # frames * (1/fps)
        ET.SubElement(spine, "asset-clip",
                      name=os.path.basename(files[idx]),
                      ref=aid,
                      start="0s",            # 媒体内部起点
                      offset=frac_to_s(start_seconds))  # 时间线偏移

    ET.ElementTree(fcpxml).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"Done: {args.out}\nFiles placed: {len(files)}")
if __name__ == "__main__":
    main()
