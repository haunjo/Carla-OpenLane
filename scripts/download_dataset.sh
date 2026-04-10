#!/bin/bash

# Carla-OpenLane Dataset Download Script
# Downloads pre-annotated CARLA datasets from Google Drive

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
SUBSET="A"
OUTPUT_DIR="../datasets/raw"

# Google Drive file IDs (UPDATE THESE WITH ACTUAL LINKS)
SUBSET_A_ID="YOUR_SUBSET_A_FILE_ID"
SUBSET_B_ID="YOUR_SUBSET_B_FILE_ID"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to download from Google Drive
download_from_gdrive() {
    local file_id=$1
    local output_file=$2

    print_info "Downloading from Google Drive (File ID: ${file_id})..."

    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        print_warning "gdown not found. Installing..."
        pip install gdown
    fi

    # Download file
    gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_file}"

    if [ $? -ne 0 ]; then
        print_error "Download failed!"
        exit 1
    fi

    print_info "Download completed: ${output_file}"
}

# Function to extract dataset
extract_dataset() {
    local archive=$1
    local output_dir=$2

    print_info "Extracting dataset to ${output_dir}..."

    # Create output directory
    mkdir -p "${output_dir}"

    # Extract based on file extension
    if [[ $archive == *.tar.gz ]] || [[ $archive == *.tgz ]]; then
        tar -xzf "${archive}" -C "${output_dir}"
    elif [[ $archive == *.tar ]]; then
        tar -xf "${archive}" -C "${output_dir}"
    elif [[ $archive == *.zip ]]; then
        unzip -q "${archive}" -d "${output_dir}"
    else
        print_error "Unsupported archive format: ${archive}"
        exit 1
    fi

    if [ $? -ne 0 ]; then
        print_error "Extraction failed!"
        exit 1
    fi

    print_info "Extraction completed!"
}

# Function to verify dataset
verify_dataset() {
    local dataset_dir=$1

    print_info "Verifying dataset integrity..."

    # Check for train and val directories
    if [ ! -d "${dataset_dir}/train" ]; then
        print_error "Missing train directory!"
        return 1
    fi

    if [ ! -d "${dataset_dir}/val" ]; then
        print_error "Missing val directory!"
        return 1
    fi

    # Count scenes
    local train_scenes=$(find "${dataset_dir}/train" -maxdepth 1 -type d | tail -n +2 | wc -l)
    local val_scenes=$(find "${dataset_dir}/val" -maxdepth 1 -type d | tail -n +2 | wc -l)

    print_info "Found ${train_scenes} training scenes and ${val_scenes} validation scenes"

    # Basic sanity check
    if [ $train_scenes -eq 0 ] || [ $val_scenes -eq 0 ]; then
        print_error "Dataset appears to be incomplete!"
        return 1
    fi

    print_info "Dataset verification passed!"
    return 0
}

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --subset A|B          Download Subset A (ArgoVerse2) or B (nuScenes) [default: A]"
    echo "  --output DIR          Output directory [default: ../datasets/raw]"
    echo "  --skip-extract        Download only, do not extract"
    echo "  --keep-archive        Keep the downloaded archive after extraction"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --subset A                    # Download Subset A"
    echo "  $0 --subset B --output /data     # Download Subset B to /data"
    exit 1
}

SKIP_EXTRACT=false
KEEP_ARCHIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --keep-archive)
            KEEP_ARCHIVE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate subset
if [[ ! "$SUBSET" =~ ^[AB]$ ]]; then
    print_error "Invalid subset: ${SUBSET}. Must be A or B."
    exit 1
fi

# Main execution
echo ""
echo "======================================"
echo " Carla-OpenLane Dataset Downloader"
echo "======================================"
echo ""
print_info "Configuration:"
echo "  Subset: ${SUBSET}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Set file ID based on subset
if [ "$SUBSET" == "A" ]; then
    FILE_ID="${SUBSET_A_ID}"
    ARCHIVE_NAME="Carla-OpenLane-subset-A.tar.gz"
    EXPECTED_SIZE="~36GB"
else
    FILE_ID="${SUBSET_B_ID}"
    ARCHIVE_NAME="Carla-OpenLane-subset-B.tar.gz"
    EXPECTED_SIZE="~32GB"
fi

# Check if file ID is set
if [ "$FILE_ID" == "YOUR_SUBSET_A_FILE_ID" ] || [ "$FILE_ID" == "YOUR_SUBSET_B_FILE_ID" ]; then
    print_error "Google Drive file IDs are not configured!"
    print_error "Please update the FILE_ID variables in this script with actual Google Drive links."
    print_error ""
    print_error "Alternatively, download manually:"
    print_error "  Subset A: https://drive.google.com/file/d/YOUR_SUBSET_A_LINK"
    print_error "  Subset B: https://drive.google.com/file/d/YOUR_SUBSET_B_LINK"
    exit 1
fi

print_warning "This will download ${EXPECTED_SIZE} of data. Ensure you have sufficient disk space."
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Download cancelled."
    exit 0
fi

# Download
TEMP_DIR=$(mktemp -d)
ARCHIVE_PATH="${TEMP_DIR}/${ARCHIVE_NAME}"

download_from_gdrive "${FILE_ID}" "${ARCHIVE_PATH}"

# Extract
if [ "$SKIP_EXTRACT" = false ]; then
    extract_dataset "${ARCHIVE_PATH}" "${OUTPUT_DIR}"

    # Verify
    verify_dataset "${OUTPUT_DIR}"

    if [ $? -ne 0 ]; then
        print_error "Dataset verification failed. Please check the downloaded files."
        exit 1
    fi

    # Cleanup
    if [ "$KEEP_ARCHIVE" = false ]; then
        print_info "Removing archive..."
        rm -rf "${TEMP_DIR}"
    else
        print_info "Archive kept at: ${ARCHIVE_PATH}"
    fi
else
    print_info "Archive saved at: ${ARCHIVE_PATH}"
fi

echo ""
print_info "✓ Dataset download completed successfully!"
print_info ""
print_info "Next steps:"
echo "  1. Verify dataset: cd ${OUTPUT_DIR} && ls -la"
echo "  2. Generate splits: cd scripts && python generate_splits.py"
echo "  3. Prepare training: cd LaneSegNet/data && python gt_generator.py"
echo ""
