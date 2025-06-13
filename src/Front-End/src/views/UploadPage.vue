<template>
  <div class="page upload-page">
    <div class="container">
      <h1 class="page-title">Upload Medical Image</h1>
      <p class="page-subtitle">Please upload a clear, high-resolution medical image for analysis</p>

      <div 
        class="upload-area" 
        :class="{ 'dragging': isDragging, 'has-file': selectedFile }"
        @dragover.prevent="handleDragOver"
        @dragleave="handleDragLeave"
        @drop.prevent="handleDrop"
        @click="!selectedFile && $refs.fileInput.click()"
      >
        <div v-if="!selectedFile" class="upload-content">
          <div class="upload-icon">
            <UploadIcon />
          </div>
          <h3 class="upload-title">Drag and drop your image here or click to browse</h3>
          <p class="upload-subtitle">Supports JPG, PNG, DICOM, and other medical image formats</p>
        </div>

        <div v-if="selectedFile" class="file-preview">
          <div class="file-info">
            <div class="success-message">File selected:</div>
            <div class="file-name">{{ selectedFile.name }}</div>
            <img v-if="previewUrl" :src="previewUrl" alt="Preview" class="preview-image" />
            <button class="remove-file-btn" @click.stop="removeFile">Remove file</button>
          </div>
        </div>

        <input 
          ref="fileInput" 
          type="file" 
          accept="image/*,.dcm" 
          @change="handleFileSelect" 
          style="display: none;"
        />
      </div>

      <FormatTags />

      <div v-if="selectedFile" class="analyze-section">
        <button 
          class="btn-analyze" 
          @click="analyzeImage" 
          :disabled="isAnalyzing"
        >
          {{ isAnalyzing ? 'Analyse en cours...' : 'Analyser le scan' }}
        </button>
      </div>

      <!-- Loading overlay -->
      <div v-if="isAnalyzing" class="loading-overlay">
        <div class="loading-content">
          <div class="spinner"></div>
          <p>Analyse de l'image en cours...</p>
          <p class="loading-subtext">Cela peut prendre quelques instants</p>
          <div v-if="uploadProgress > 0" class="progress-container">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
            </div>
            <span class="progress-text">{{ uploadProgress }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import UploadIcon from '../components/icons/UploadIcon.vue'
import FormatTags from '../components/FormatTags.vue'

export default {
  name: 'UploadPage',
  components: { UploadIcon, FormatTags },
  data() {
    return {
      selectedFile: null,
      previewUrl: null,
      isDragging: false,
      isAnalyzing: false,
      uploadProgress: 0
    }
  },
  methods: {
    handleDragOver(e) {
      e.preventDefault(); this.isDragging = true;
    },
    handleDragLeave() {
      this.isDragging = false;
    },
    handleDrop(e) {
      e.preventDefault(); this.isDragging = false;
      if (e.dataTransfer.files.length) this.setSelectedFile(e.dataTransfer.files[0]);
    },
    handleFileSelect(e) {
      if (e.target.files.length) this.setSelectedFile(e.target.files[0]);
    },
    setSelectedFile(file) {
      this.selectedFile = file;
      if (file.type.startsWith('image/')) this.previewUrl = URL.createObjectURL(file);
    },
    removeFile() {
      this.selectedFile = null;
      this.previewUrl = null;
      this.uploadProgress = 0;
      this.$refs.fileInput.value = '';
    },
    async analyzeImage() {
      if (!this.selectedFile) return;
      this.isAnalyzing = true;
      this.uploadProgress = 0;

      try {
        const formData = new FormData();
        formData.append('file', this.selectedFile);

        const config = {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 300000,
          onUploadProgress: e => {
            if (e.lengthComputable) this.uploadProgress = Math.round((e.loaded * 100) / e.total);
          }
        };

        const { data } = await axios.post('/api/analyze-scan', formData, config);
        this.$router.push({ name: 'results', params: { id: data.analysisId } });
      } catch (err) {
        console.error(err);
        alert('Une erreur est survenue lors de l’analyse.');
      } finally {
        this.isAnalyzing = false;
        this.uploadProgress = 0;
      }
    }
  },
  beforeUnmount() {
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
  }
}
</script>

<style scoped>
.page-title {
  font-size: 36px;
  font-weight: bold;
  text-align: center;
  color: #7c3aed;
  margin-bottom: 24px;
}

.page-subtitle {
  font-size: 20px;
  text-align: center;
  color: #6b7280;
  margin-bottom: 40px;
}

.upload-area {
  border: 2px dashed #c4b5fd;
  border-radius: 24px;
  padding: 48px;
  text-align: center;
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 24px;
}

.upload-area.dragging {
  border-color: #7c3aed;
  background: #f5f3ff;
}

.upload-area.has-file {
  cursor: default;
}

.upload-icon {
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, #3b82f6, #7c3aed);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 24px;
  box-shadow: 0 8px 16px rgba(124, 58, 237, 0.3);
  color: white;
}

.upload-title {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #374151;
}

.upload-subtitle {
  color: #6b7280;
  margin-bottom: 24px;
}

.file-preview {
  text-align: center;
}

.success-message {
  color: #059669;
  font-weight: 600;
  margin-bottom: 8px;
}

.file-name {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 24px;
}

.preview-image {
  max-height: 256px;
  max-width: 100%;
  border-radius: 8px;
  margin-bottom: 24px;
}

.remove-file-btn {
  color: #dc2626;
  background: none;
  border: none;
  text-decoration: underline;
  cursor: pointer;
  font-size: 14px;
}

.analyze-section {
  text-align: center;
}

.btn-analyze {
  background: #7c3aed;
  color: white;
  border: none;
  padding: 16px 48px;
  border-radius: 50px;
  font-size: 18px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
}

.btn-analyze:hover:not(:disabled) {
  background: #6d28d9;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(124, 58, 237, 0.4);
}

.btn-analyze:disabled {
  background: #c4b5fd;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.loading-content {
  background: white;
  padding: 48px;
  border-radius: 16px;
  text-align: center;
  max-width: 400px;
  margin: 0 20px;
}

.spinner {
  width: 60px;
  height: 60px;
  border: 6px solid #f3f3f3;
  border-top: 6px solid #7c3aed;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 24px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-content p {
  font-size: 18px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 8px;
}

.loading-subtext {
  font-size: 14px !important;
  color: #6b7280 !important;
  font-weight: normal !important;
  margin-bottom: 16px !important;
}

.progress-container {
  margin-top: 16px;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 8px;
}

.progress-fill {
  height: 100%;
  background: #7c3aed;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 14px;
  color: #6b7280;
  font-weight: 500;
}

@media (max-width: 768px) {
  .upload-area {
    padding: 24px;
    min-height: 300px;
  }
  
  .loading-content {
    padding: 32px 24px;
  }
}
</style>

