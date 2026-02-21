import { X, Upload, Camera, Loader2 } from 'lucide-react';
import { useState } from 'react';

export default function UploadModal({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) {
  if (!isOpen) return null;

  const [isUploading, setIsUploading] = useState(false);

  const handleUpload = () => {
    setIsUploading(true);
    // This is where we will eventually fetch to your Python /app.py
    setTimeout(() => {
      setIsUploading(false);
      onClose();
    }, 2000);
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-[#111111] border border-white/10 w-full max-w-md rounded-3xl p-8 relative">
        <button onClick={onClose} className="absolute right-6 top-6 text-zinc-500 hover:text-white">
          <X size={20} />
        </button>

        <h2 className="text-2xl font-bold mb-2">New Inventory Scan</h2>
        <p className="text-zinc-500 text-sm mb-8">Upload a site photo to analyze inventory levels via AI.</p>

        <div className="space-y-4">
          <div className="border-2 border-dashed border-white/5 rounded-2xl p-10 flex flex-col items-center justify-center group hover:border-yellow-500/50 transition-colors cursor-pointer">
            <Upload className="text-zinc-700 group-hover:text-yellow-500 mb-4 transition-colors" size={40} />
            <p className="text-sm font-medium text-zinc-400">Drag & drop or <span className="text-yellow-500">browse</span></p>
          </div>

          <button 
            onClick={handleUpload}
            disabled={isUploading}
            className="w-full bg-yellow-500 hover:bg-yellow-400 text-black font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            {isUploading ? <Loader2 className="animate-spin" /> : <Camera size={20} />}
            {isUploading ? 'Analyzing...' : 'Start AI Audit'}
          </button>
        </div>
      </div>
    </div>
  );
}