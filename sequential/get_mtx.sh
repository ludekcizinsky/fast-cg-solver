#!/bin/bash

# Check for required argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <MatrixMarket URL to .tar.gz file>"
    exit 1
fi

# Create matrices directory if it doesn't exist
mkdir -p matrices

# Extract filename from URL
URL="$1"
FILENAME=$(basename "$URL")              # e.g., CurlCurl_2.tar.gz
BASENAME="${FILENAME%.tar.gz}"           # e.g., CurlCurl_2

# Download into matrices/
wget -q --show-progress -O "matrices/$FILENAME" "$URL"

# Extract into a temporary subdirectory
tar -xzf "matrices/$FILENAME" -C matrices

# Move any .mtx files from extracted folder to matrices/
find "matrices/$BASENAME" -name "*.mtx" -exec mv {} matrices/ \;

# Optionally, remove the extracted folder
rm -rf "matrices/$BASENAME"

# Remove the tar.gz file
rm "matrices/$FILENAME"

echo "✅ Done: Matrix extracted to matrices/, tarball deleted."

# Run the cgsolver
./cgsolver "matrices/$BASENAME.mtx"