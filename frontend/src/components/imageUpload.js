import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, ImageIcon, Camera, Loader } from 'lucide-react';
import '../assets/styles/imageUpload.css';
import { io } from 'socket.io-client';

const socket = io('http://localhost:5050');

const ImageUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [image, setImage] = useState(null);
  const [isVisible, setIsVisible] = useState(false);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

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
  const handleClose = () => {
    setIsVisible(false);
    setImage(null); 
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

  useEffect(() => {
    // Nettoyer le flux de la caméra quand le composant est démonté
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    // Démarrer la caméra une fois que isCapturing est true et que videoRef est disponible
    if (isCapturing && videoRef.current) {
      const startVideoStream = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            streamRef.current = stream;
          }
        } catch (err) {
          setError('Erreur lors de l\'accès à la caméra');
          console.error('Erreur caméra:', err);
          setIsCapturing(false);
        }
      };

      startVideoStream();
    }
  }, [isCapturing]);

  const startCamera = () => {
    setIsCapturing(true);
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCapturing(false);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;

    const canvas = document.createElement('canvas');
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    canvas.toBlob(blob => {
      const file = new File([blob], `photo_${Date.now()}.jpg`, { type: 'image/jpeg' });
      setSelectedFiles(prev => [...prev, file]);
    }, 'image/jpeg');
    
    stopCamera();
  };

  const handleSubmit = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      console.log('Aucun fichier sélectionné');
      return;
    }
  
    const formData = new FormData();
    formData.append('image', selectedFiles[0]); // On prend le premier fichier sélectionné
    
    setIsLoading(true);
    try {
      // Envoi du fichier à votre API Flask
      const response = await fetch('http://localhost:5050/upload', {
        method: 'POST',
        body: formData
      });
  
      if (!response.ok) {
        throw new Error('Erreur lors de l\'upload');
      }
  
      const data = await response.json();
      
      // Une fois l'image enregistrée, on peut émettre l'événement socket
      // avec le chemin retourné par l'API
      socket.emit('process_image', { path: data.path });
  
    } catch (error) {
      console.error('Erreur:', error);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    socket.on("response", (response) => {
        if (response.image) {
            setImage(response.image);
            setIsVisible(true);
            setIsLoading(false);
        }
    });

    return () => socket.off("response");
}, []);

  return (

    <div className="image-upload-container">
      <h1 className="upload-title">Dépose un photo de toi ici :</h1>
      {isVisible && (
        <div id="popup">
           <button id="close-btn" onClick={handleClose}>✖</button>
          <h2>Vous ressemblez à Cette Star :</h2>
          {image && <img src={image} alt="Processed" />}
        </div>
      )}
      <div className="center-container">
        {isCapturing ? (
          <div className="camera-container">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline
              className="camera-preview"
            />
            <div className="camera-controls">
              <button onClick={capturePhoto} className="capture-button">
                Prendre la photo
              </button>
              
              <button onClick={stopCamera} className="cancel-button">
                Annuler
              </button>
            </div>
          </div>
        ) : (
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
            <button onClick={startCamera} className="camera-button">
              <Camera className="camera-icon" />
              Prendre une photo
            </button>
          </div>
        )}

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
              disabled={isLoading}
            >
              {isLoading ? <Loader className="loader-icon" /> : 'Trouver la star qui se cache en moi !'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;