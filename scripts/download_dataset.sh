#!/bin/bash
# Carla-OpenLane — download dataset and annotation tool
#
# Usage:
#   ./scripts/download_dataset.sh --subset A        # dataset only
#   ./scripts/download_dataset.sh --subset B
#   ./scripts/download_dataset.sh --converter-only  # annotation tool only
#   ./scripts/download_dataset.sh --all             # dataset + tool

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Google Drive file IDs (fill in after uploading) ─────────────────────────
SUBSET_A_GDRIVE_ID="YOUR_SUBSET_A_FILE_ID"
SUBSET_B_GDRIVE_ID="YOUR_SUBSET_B_FILE_ID"

# ── GitHub release asset URL ─────────────────────────────────────────────────
CONVERTER_RELEASE_URL="https://github.com/haunjo/Carla-OpenLane/releases/download/v1.1/OpenLane-V2-HDmap-Converter-v1.0.zip"

# ── Defaults ─────────────────────────────────────────────────────────────────
SUBSET=""
OUTPUT_DIR="$(dirname "$0")/../datasets/raw"
DOWNLOAD_CONVERTER=false
DOWNLOAD_DATASET=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  --subset A|B          Download dataset (Subset A or B)"
    echo "  --converter-only      Download annotation tool only"
    echo "  --all                 Download dataset (Subset B) + annotation tool"
    echo "  --output DIR          Dataset output directory [default: datasets/raw]"
    echo "  -h, --help            Show this help"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset)         SUBSET="$2"; DOWNLOAD_DATASET=true; shift 2 ;;
        --converter-only) DOWNLOAD_CONVERTER=true; shift ;;
        --all)            DOWNLOAD_DATASET=true; SUBSET="B"; DOWNLOAD_CONVERTER=true; shift ;;
        --output)         OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) print_error "Unknown option: $1"; usage ;;
    esac
done

if [[ "$DOWNLOAD_DATASET" == false && "$DOWNLOAD_CONVERTER" == false ]]; then
    usage
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
require_cmd() {
    command -v "$1" &>/dev/null || { print_error "$1 is required but not installed."; exit 1; }
}

download_gdrive() {
    local file_id=$1 output=$2
    require_cmd gdown
    print_info "Downloading from Google Drive..."
    gdown "https://drive.google.com/uc?id=${file_id}" -O "$output"
}

# ── Download annotation converter ────────────────────────────────────────────
download_converter() {
    local dest="$(dirname "$0")/../OpenLane-V2-HDmap-Converter"
    if [[ -d "$dest" ]]; then
        print_info "OpenLane-V2-HDmap-Converter already exists at $dest — skipping."
        return
    fi

    print_info "Downloading OpenLane-V2-HDmap-Converter..."
    local tmp=$(mktemp -d)

    if command -v curl &>/dev/null; then
        curl -L "$CONVERTER_RELEASE_URL" -o "$tmp/converter.zip"
    elif command -v wget &>/dev/null; then
        wget -q "$CONVERTER_RELEASE_URL" -O "$tmp/converter.zip"
    else
        print_error "curl or wget is required to download the converter."
        exit 1
    fi

    unzip -q "$tmp/converter.zip" -d "$tmp/extracted"
    mv "$tmp/extracted" "$dest"
    rm -rf "$tmp"

    print_info "Converter installed at: $dest"
    print_info "Docker image setup: see $dest/docker/DOCKER_DISTRIBUTION.md"
}

# ── Download dataset ──────────────────────────────────────────────────────────
download_dataset() {
    local subset=$1

    if [[ ! "$subset" =~ ^[AB]$ ]]; then
        print_error "Invalid subset '$subset'. Must be A or B."
        exit 1
    fi

    if [[ "$subset" == "A" ]]; then
        local file_id="$SUBSET_A_GDRIVE_ID"
        local archive="Carla-OpenLane-subset-A.tar.gz"
        local size="~36 GB"
    else
        local file_id="$SUBSET_B_GDRIVE_ID"
        local archive="Carla-OpenLane-subset-B.tar.gz"
        local size="~32 GB"
    fi

    if [[ "$file_id" == YOUR_* ]]; then
        print_error "Google Drive file ID not configured for Subset ${subset}."
        print_error "Update SUBSET_${subset}_GDRIVE_ID in this script after uploading the dataset."
        exit 1
    fi

    print_warning "Downloading Subset ${subset} (${size}). Ensure you have sufficient disk space."
    read -p "Continue? (y/n) " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && { print_info "Cancelled."; exit 0; }

    mkdir -p "$OUTPUT_DIR"
    local tmp=$(mktemp -d)
    download_gdrive "$file_id" "$tmp/$archive"

    print_info "Extracting to $OUTPUT_DIR ..."
    tar -xzf "$tmp/$archive" -C "$OUTPUT_DIR"
    rm -rf "$tmp"

    print_info "Dataset extracted to: $OUTPUT_DIR"
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo " Carla-OpenLane Setup"
echo "======================================"
echo ""

[[ "$DOWNLOAD_CONVERTER" == true ]] && download_converter
[[ "$DOWNLOAD_DATASET" == true ]]   && download_dataset "$SUBSET"

echo ""
print_info "Done. See docs/FULL_WORKFLOW.md for next steps."
