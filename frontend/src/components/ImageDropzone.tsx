import React, { useRef, useState } from 'react';
import { UploadCloud, X } from 'lucide-react';

interface ImageDropzoneProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  onClear: () => void;
  label?: string;
  accentColor?: 'blue' | 'teal' | 'indigo';
}

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({
  selectedFile,
  previewUrl,
  onFileSelect,
  onClear,
  label = 'Upload Site Image',
  accentColor = 'blue',
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const borderClass = {
    blue: 'hover:border-blue-500 hover:bg-blue-50/50',
    teal: 'hover:border-teal-500 hover:bg-teal-50/50',
    indigo: 'hover:border-indigo-500 hover:bg-indigo-50/50',
  }[accentColor];

  const activeBorderClass = {
    blue: 'border-blue-500 bg-blue-50/60',
    teal: 'border-teal-500 bg-teal-50/60',
    indigo: 'border-indigo-500 bg-indigo-50/60',
  }[accentColor];

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragOver(true);
    } else if (e.type === 'dragleave') {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelect(file);
      } else {
        alert('Please choose a valid image file (.jpg, .png, .webp).');
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelect(file);
      }
    }
  };

  return (
    <div>
      <label className="block text-xs font-semibold text-slate-600 uppercase mb-2">
        {label}
      </label>

      <div
        onClick={() => !selectedFile && fileInputRef.current?.click()}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition flex flex-col items-center justify-center min-h-[150px] relative ${
          isDragOver ? activeBorderClass : `border-slate-300 ${borderClass}`
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />

        {previewUrl ? (
          <div className="w-full relative">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-h-52 mx-auto rounded-lg object-contain border border-slate-200 shadow-sm"
            />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onClear();
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              className="absolute top-1 right-1 bg-red-600 hover:bg-red-700 text-white rounded-full w-7 h-7 flex items-center justify-center text-xs shadow-md transition"
              title="Remove image"
            >
              <X className="w-4 h-4" />
            </button>
            <p className="text-[11px] text-slate-500 mt-2 font-mono truncate max-w-xs mx-auto">
              {selectedFile?.name}
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="w-11 h-11 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center mb-2">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-slate-700">
              Drag & drop image or <span className="text-blue-600 font-semibold underline">browse</span>
            </p>
            <p className="text-xs text-slate-400 mt-1">Supports JPG, PNG, WEBP (Max 10MB)</p>
          </div>
        )}
      </div>
    </div>
  );
};
