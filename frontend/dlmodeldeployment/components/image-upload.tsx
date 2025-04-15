"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Upload } from "lucide-react"

interface ImageUploadProps {
  onUpload: (file: File) => void
}

export function ImageUpload({ onUpload }: ImageUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
    }
  }

  const handleUpload = () => {
    if (file) {
        onUpload(file)
        setFile(null)
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid w-full max-w-sm items-center gap-1.5">
        <Label htmlFor="image">OCT Scan Image</Label>
        <Input id="image" type="file" accept="image/*" onChange={handleFileChange} ref={fileInputRef} />
      </div>
      <Button onClick={handleUpload} disabled={!file}>
        <Upload className="mr-2 h-4 w-4" /> Upload Image
      </Button>
    </div>
  )
}

