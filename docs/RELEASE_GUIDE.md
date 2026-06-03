# Release Guide (Maintainer)

This document describes how to cut a new public release of Carla-OpenLane.

---

## 1. Package the Annotation Tool

```bash
cd /path/to/OpenLane-V2-HDmap-Converter

zip -r OpenLane-V2-HDmap-Converter-v<VERSION>.zip \
  src/carla2openlanev2.py \
  src/checksum.py \
  src/repair_lste.py \
  docker/run_docker.sh \
  docker/DOCKER_DISTRIBUTION.md \
  README.md \
  README_ko.md \
  LICENSE
```

## 2. Create a GitHub Release

1. Go to `https://github.com/haunjo/Carla-OpenLane/releases/new`
2. Tag: `v<VERSION>` (e.g. `v1.0`)
3. Title: `Carla-OpenLane v<VERSION>`
4. Attach `OpenLane-V2-HDmap-Converter-v<VERSION>.zip` as a release asset
5. Publish release

After publishing, the download URL will be:
```
https://github.com/haunjo/Carla-OpenLane/releases/download/v<VERSION>/OpenLane-V2-HDmap-Converter-v<VERSION>.zip
```

Update `CONVERTER_RELEASE_URL` in `scripts/download_dataset.sh` to match.

## 3. Upload Dataset to Google Drive

Upload the dataset archives and update the file IDs in `scripts/download_dataset.sh`:

```bash
SUBSET_A_GDRIVE_ID="<Google Drive file ID for subset A>"
SUBSET_B_GDRIVE_ID="<Google Drive file ID for subset B>"
```

Archive naming convention:
- `Carla-OpenLane-subset-A.tar.gz` — Subset A (Argoverse2, ~36 GB)
- `Carla-OpenLane-subset-B.tar.gz` — Subset B (nuScenes, ~32 GB)

## 4. Push Docker Image to Docker Hub

```bash
docker tag lanelet2 haunjo/lanelet2:latest
docker tag lanelet2 haunjo/lanelet2:v<VERSION>
docker login
docker push haunjo/lanelet2:latest
docker push haunjo/lanelet2:v<VERSION>
```

Grant read access to collaborators via Docker Hub repository settings.
