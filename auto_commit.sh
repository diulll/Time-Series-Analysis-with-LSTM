#!/bin/bash

# Cek apakah ada perubahan
if [ -z "$(git status --porcelain)" ]; then 
    echo "Tidak ada perubahan untuk di-commit."
    exit 0
fi

# Loop setiap baris output dari git status --porcelain
git status --porcelain | while IFS= read -r line; do
    # Mengambil nama file dengan awk (kolom ke-2 dari output git status --porcelain)
    # Untuk file dengan spasi, git akan wrap dengan quotes, kita handle itu
    file=$(echo "$line" | awk '{print substr($0, 4)}')
    
    # Hapus tanda kutip jika ada (untuk file dengan spasi)
    file=$(echo "$file" | sed 's/^"//;s/"$//')

    if [ -n "$file" ]; then
        echo "================================"
        echo "Committing: $file"
        
        # Add file spesifik
        git add "$file"
        
        # Commit dengan pesan nama file
        git commit -m "Update/Add $file"
        echo "✓ Done"
    fi
done

echo "============================================="
echo "Selesai! Semua file telah di-commit terpisah."
echo "============================================="
