#!/usr/bin/env python
import argparse
import os
import sys
import mimetypes
import boto3
from botocore.config import Config


def is_mesh_file(path: str) -> bool:
    mesh_exts = {".obj", ".mtl", ".glb", ".gltf", ".ply", ".fbx", ".png", ".jpg", ".jpeg"}
    _, ext = os.path.splitext(path.lower())
    return ext in mesh_exts


def iter_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            full = os.path.join(dirpath, f)
            if is_mesh_file(full):
                yield full


def main():
    p = argparse.ArgumentParser(description="Upload mesh outputs to S3")
    p.add_argument("--source", required=True, help="Local directory containing meshes (e.g., /workspace/output/run1/mesh)")
    p.add_argument("--s3_url", required=True, help="S3 URL prefix, e.g., s3://bucket/key/prefix or https://<bucket>.s3.<region>.amazonaws.com/key/prefix/")
    p.add_argument("--acl", default="public-read", help="Canned ACL for uploaded objects (default: public-read)")
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-1")
    args = p.parse_args()

    src = os.path.abspath(args.source)
    if not os.path.isdir(src):
        print(f"Source directory not found: {src}", file=sys.stderr)
        sys.exit(2)

    # Parse s3_url into bucket and key prefix
    url = args.s3_url
    bucket = None
    key_prefix = ""
    if url.startswith("s3://"):
        rest = url[len("s3://"):]
        parts = rest.split("/", 1)
        bucket = parts[0]
        key_prefix = parts[1] if len(parts) > 1 else ""
    elif "amazonaws.com" in url:
        # e.g., https://bucket.s3.region.amazonaws.com/prefix/
        host_and_path = url.split("//", 1)[-1]
        host, _, path = host_and_path.partition("/")
        # host could be bucket.s3.region.amazonaws.com
        bucket = host.split(".")[0]
        key_prefix = path
    else:
        print("Unsupported s3_url format. Use s3://bucket/prefix or https://bucket.s3.region.amazonaws.com/prefix/", file=sys.stderr)
        sys.exit(2)

    key_prefix = key_prefix.strip("/")

    session = boto3.session.Session(region_name=args.region)
    s3 = session.client("s3", config=Config(s3={'addressing_style': 'virtual'}))

    uploaded = 0
    for f in iter_files(src):
        rel = os.path.relpath(f, src).replace(os.sep, "/")
        key = f"{key_prefix}/{rel}" if key_prefix else rel
        ctype, _ = mimetypes.guess_type(f)
        extra = {"ACL": args.acl}
        if ctype:
            extra["ContentType"] = ctype
        print(f"Uploading {f} -> s3://{bucket}/{key}")
        s3.upload_file(f, bucket, key, ExtraArgs=extra)
        uploaded += 1

    print(f"Done. Uploaded {uploaded} files to s3://{bucket}/{key_prefix}")


if __name__ == "__main__":
    main()


