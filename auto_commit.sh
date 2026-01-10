#!/bin/bash

# Cek apakah ada perubahan
if [ -z "$(git status --porcelain)" ]; then 
    echo "Tidak ada perubahan untuk di-commit."
    exit 0
fi

# Loop setiap baris output dari git status --porcelain
git status --porcelain | while read -r line; do
    # Mengambil nama file (hapus 3 karakter pertama yang merupakan status git)
    file=$(echo "$line" | sed 's/^...//')
    
    # Hapus tanda kutip jika ada (untuk file dengan spasi)
    file=$(echo "$file" | sed 's/^"//;s/"$//')

    if [ -n "$file" ]; then
        echo "Committing: $file"
        
        # Add file spesifik
        git add "$file"
        
        # Commit dengan pesan nama file
        git commit -m "Update/Add $file"
    fi
done

echo "============================================="
echo "Selesai! Semua file tekah di-commit terpisah."
echo "============================================="
