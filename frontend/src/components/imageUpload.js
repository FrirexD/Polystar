import React, { useState } from 'react';
import { Upload, X, ImageIcon } from 'lucide-react';
import '../assets/styles/imageUpload.css';

const ImageUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState('');

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = [...e.dataTransfer.files];
    handleFiles(files);
  };

  const handleFileSelect = (e) => {
    const files = [...e.target.files];
    handleFiles(files);
  };

  const handleFiles = (files) => {
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    if (files.length !== imageFiles.length) {
      setError('Veuillez ne sélectionner que des images');
      return;
    }
    setError('');
    setSelectedFiles(prev => [...prev, ...imageFiles]);
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    console.log('Files to upload:', selectedFiles);
  };

  return (
    <div className="image-upload-container">
      <h1 className="upload-title">Dépose un photo de toi ici :</h1>
      <div className="center-container">
        <div
          className={`drop-zone ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <input
            type="file"
            multiple
            accept="image/*"
            className="hidden"
            id="file-upload"
            onChange={handleFileSelect}
          />
          <label htmlFor="file-upload" className="upload-label">
            <Upload className="upload-icon" />
            <p className="upload-text">
              Déposez vos images ici ou <span>parcourir</span>
            </p>
          </label>
        </div>

        {error && <div className="error-message">{error}</div>}

        {selectedFiles.length > 0 && (
          <div className="file-list">
            <div className="file-grid">
              {selectedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <ImageIcon className="file-icon" />
                  <div className="file-details">
                    <p className="file-name">{file.name}</p>
                    <p className="file-size">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="remove-button"
                  >
                    <X className="remove-icon" />
                  </button>
                </div>
              ))}
            </div>
            <button
              className="upload-button"
              onClick={handleSubmit}
            >
              Télécharger {selectedFiles.length} image{selectedFiles.length > 1 ? 's' : ''}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;