<template>
  <div class="page results-page">
    <div class="container">
      <router-link to="/upload" class="back-btn">
        <BackIcon />
        Retour à l'upload
      </router-link>

      <h1 class="page-title">Résultats de l'analyse</h1>

      <div v-if="loading" class="loading-container">
        <div class="spinner"></div>
        <p>Chargement des résultats...</p>
      </div>

      <div v-else-if="error" class="error-container">
        <h2>Erreur</h2>
        <p>{{ error }}</p>
        <div class="error-actions">
          <button @click="fetchResults" class="btn-retry">Réessayer</button>
          <router-link to="/upload" class="btn-primary">Nouvelle analyse</router-link>
        </div>
      </div>

      <template v-else>
        <!-- Action buttons -->
        <div class="action-buttons">
          <button @click="downloadReport" class="btn-download" :disabled="downloadingReport">
            <DownloadIcon />
            {{ downloadingReport ? 'Génération...' : 'Télécharger le rapport' }}
          </button>
          <button @click="shareResults" class="btn-share">
            <ShareIcon />
            Partager
          </button>
        </div>

        <!-- Results Grid -->
        <div class="results-grid">
          <div class="result-card">
            <h2 class="result-title">Scan Original</h2>
            <div class="image-container">
              <img :src="analysisResult.originalScan" alt="Scan médical original" class="result-image" />
              <div class="image-info">
                <span class="image-size">{{ analysisResult.originalSize }}</span>
                <span class="image-format">{{ analysisResult.originalFormat }}</span>
              </div>
            </div>
          </div>

          <div class="result-card">
            <h2 class="result-title">Scan Généré (IA)</h2>
            <div class="image-container">
              <img :src="analysisResult.generatedScan" alt="Scan généré par IA" class="result-image" />
              <div class="image-info">
                <span class="enhancement-type">{{ analysisResult.enhancementType }}</span>
                <span class="confidence">Confiance: {{ analysisResult.confidence }}%</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Analysis Summary -->
        <div class="analysis-summary">
          <h2 class="summary-title">Résumé de l'analyse</h2>
          <div class="summary-content">
            <div class="summary-item">
              <span class="summary-label">Date d'analyse:</span>
              <span class="summary-value">{{ formatDate(analysisResult.analysisDate) }}</span>
            </div>
            <div class="summary-item">
              <span class="summary-label">Durée du traitement:</span>
              <span class="summary-value">{{ analysisResult.processingTime }}s</span>
            </div>
            <div class="summary-item">
              <span class="summary-label">Modèle IA utilisé:</span>
              <span class="summary-value">{{ analysisResult.aiModel }}</span>
            </div>
          </div>
        </div>

        <!-- Predictions -->
        <div class="predictions-card">
          <h2 class="predictions-title">Prédictions de l'IA</h2>
          <div class="predictions-list">
            <div 
              v-for="(prediction, index) in analysisResult.predictions" 
              :key="index" 
              class="prediction-item"
              :class="{ 'high-confidence': prediction.probability > 0.7 }"
            >
              <div class="prediction-header">
                <span class="prediction-label">{{ prediction.label }}</span>
                <span class="prediction-percentage">{{ (prediction.probability * 100).toFixed(1) }}%</span>
              </div>
              <div class="progress-bar">
                <div 
                  class="progress-fill" 
                  :style="{ width: (prediction.probability * 100) + '%' }"
                ></div>
              </div>
              <p v-if="prediction.description" class="prediction-description">
                {{ prediction.description }}
              </p>
            </div>
          </div>
        </div>

        <!-- Recommendations -->
        <div v-if="analysisResult.recommendations" class="recommendations-card">
          <h2 class="recommendations-title">Recommandations</h2>
          <ul class="recommendations-list">
            <li v-for="(recommendation, index) in analysisResult.recommendations" :key="index">
              {{ recommendation }}
            </li>
          </ul>
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer">
          <h3 class="disclaimer-title">Note importante</h3>
          <p class="disclaimer-text">
            Cette analyse est fournie uniquement comme aide au diagnostic et ne doit pas remplacer 
            l'avis médical professionnel. Veuillez consulter un professionnel de la santé pour 
            un diagnostic et un traitement appropriés.
          </p>
        </div>
      </template>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import BackIcon from '../components/icons/BackIcon.vue'
import DownloadIcon from '../components/icons/DownloadIcon.vue'
import ShareIcon from '../components/icons/ShareIcon.vue'

export default {
  name: 'ResultsPage',
  components: {
    BackIcon,
    DownloadIcon,
    ShareIcon
  },
  props: {
    id: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      loading: true,
      error: null,
      analysisResult: null,
      downloadingReport: false
    }
  },
  created() {
    this.fetchResults()
  },
  watch: {
    // Watch for route changes to refetch data
    '$route'() {
      this.fetchResults()
    }
  },
  methods: {
    async fetchResults() {
      this.loading = true
      this.error = null
      
      try {
        // Configure axios request with timeout
        const config = {
          timeout: 30000, // 30 seconds timeout
          headers: {
            'Accept': 'application/json'
          }
        }
        
        // Fetch analysis results from backend API using axios
        const response = await axios.get(`/api/analysis-results/${this.id}`, config)
        
        this.analysisResult = response.data
        
      } catch (error) {
        console.error('Error fetching results:', error)
        
        let errorMessage = 'Impossible de charger les résultats de l\'analyse.'
        
        if (error.response) {
          // Server responded with error status
          switch (error.response.status) {
            case 404:
              errorMessage = 'Analyse non trouvée. Elle a peut-être expiré.'
              break
            case 403:
              errorMessage = 'Accès non autorisé à cette analyse.'
              break
            case 500:
              errorMessage = 'Erreur serveur. Veuillez réessayer plus tard.'
              break
            default:
              errorMessage = `Erreur ${error.response.status}: ${error.response.data?.message || 'Erreur inconnue'}`
          }
        } else if (error.request) {
          // Network error
          errorMessage = 'Erreur de connexion. Vérifiez votre connexion internet.'
        } else if (error.code === 'ECONNABORTED') {
          // Timeout error
          errorMessage = 'Le chargement prend trop de temps. Veuillez réessayer.'
        }
        
        this.error = errorMessage
      } finally {
        this.loading = false
      }
    },
    
    async downloadReport() {
      this.downloadingReport = true
      
      try {
        // Configure axios request for file download
        const config = {
          responseType: 'blob',
          timeout: 60000, // 1 minute timeout for report generation
          headers: {
            'Accept': 'application/pdf'
          }
        }
        
        const response = await axios.post(`/api/generate-report/${this.id}`, {
          format: 'pdf',
          includeImages: true,
          language: 'fr'
        }, config)
        
        // Create download link
        const blob = new Blob([response.data], { type: 'application/pdf' })
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        
        // Get filename from response headers or use default
        const contentDisposition = response.headers['content-disposition']
        let filename = `rapport-analyse-${this.id}.pdf`
        
        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename="(.+)"/)
          if (filenameMatch) {
            filename = filenameMatch[1]
          }
        }
        
        link.download = filename
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
        
      } catch (error) {
        console.error('Error downloading report:', error)
        
        let errorMessage = 'Erreur lors du téléchargement du rapport. Veuillez réessayer.'
        
        if (error.response) {
          switch (error.response.status) {
            case 404:
              errorMessage = 'Rapport non disponible pour cette analyse.'
              break
            case 429:
              errorMessage = 'Trop de demandes de rapport. Veuillez patienter.'
              break
            case 500:
              errorMessage = 'Erreur lors de la génération du rapport.'
              break
          }
        } else if (error.code === 'ECONNABORTED') {
          errorMessage = 'La génération du rapport prend trop de temps. Veuillez réessayer.'
        }
        
        alert(errorMessage)
      } finally {
        this.downloadingReport = false
      }
    },
    
    shareResults() {
      const shareData = {
        title: 'Résultats d\'analyse MediScan',
        text: 'Consultez les résultats de mon analyse médicale',
        url: window.location.href
      }
      
      if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
        navigator.share(shareData).catch(error => {
          console.error('Error sharing:', error)
          this.fallbackShare()
        })
      } else {
        this.fallbackShare()
      }
    },
    
    fallbackShare() {
      // Fallback: copy to clipboard
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(window.location.href).then(() => {
          alert('Lien copié dans le presse-papiers!')
        }).catch(() => {
          this.manualCopyFallback()
        })
      } else {
        this.manualCopyFallback()
      }
    },
    
    manualCopyFallback() {
      // Manual copy fallback for older browsers
      const textArea = document.createElement('textarea')
      textArea.value = window.location.href
      document.body.appendChild(textArea)
      textArea.select()
      try {
        document.execCommand('copy')
        alert('Lien copié dans le presse-papiers!')
      } catch (err) {
        console.error('Failed to copy:', err)
        alert('Impossible de copier le lien automatiquement.')
      }
      document.body.removeChild(textArea)
    },
    
    formatDate(dateString) {
      const date = new Date(dateString)
      return date.toLocaleDateString('fr-FR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }
  }
}
</script>

<style scoped>
.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  background: none;
  border: none;
  color: #7c3aed;
  font-size: 16px;
  cursor: pointer;
  margin-bottom: 32px;
  text-decoration: none;
}

.back-btn:hover {
  text-decoration: underline;
}

.page-title {
  font-size: 36px;
  font-weight: bold;
  text-align: center;
  color: #7c3aed;
  margin-bottom: 32px;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-bottom: 48px;
}

.btn-download,
.btn-share {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.btn-download {
  background: #059669;
  color: white;
}

.btn-download:hover:not(:disabled) {
  background: #047857;
}

.btn-download:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

.btn-share {
  background: #3b82f6;
  color: white;
}

.btn-share:hover {
  background: #2563eb;
}

.btn-retry {
  background: #f59e0b;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  margin-right: 16px;
}

.btn-retry:hover {
  background: #d97706;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 32px;
  margin-bottom: 48px;
}

.result-card {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.result-title {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #374151;
}

.image-container {
  position: relative;
  aspect-ratio: 1;
  background: #f3f4f6;
  border-radius: 8px;
  overflow: hidden;
}

.result-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.image-info {
  position: absolute;
  bottom: 8px;
  left: 8px;
  right: 8px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 8px;
  border-radius: 4px;
  font-size: 12px;
  display: flex;
  justify-content: space-between;
}

.analysis-summary {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  margin-bottom: 32px;
}

.summary-title {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #374151;
}

.summary-content {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  padding: 12px;
  background: #f9fafb;
  border-radius: 6px;
}

.summary-label {
  font-weight: 500;
  color: #6b7280;
}

.summary-value {
  font-weight: 600;
  color: #374151;
}

.predictions-card {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  margin-bottom: 32px;
}

.predictions-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 24px;
  color: #374151;
}

.prediction-item {
  margin-bottom: 24px;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
}

.prediction-item.high-confidence {
  border-color: #f59e0b;
  background: #fffbeb;
}

.prediction-item:last-child {
  margin-bottom: 0;
}

.prediction-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.prediction-label {
  font-weight: 600;
  font-size: 18px;
  color: #374151;
}

.prediction-percentage {
  font-size: 18px;
  font-weight: bold;
  color: #7c3aed;
}

.progress-bar {
  width: 100%;
  height: 16px;
  background: #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 8px;
}

.progress-fill {
  height: 100%;
  background: #7c3aed;
  border-radius: 8px;
  transition: width 0.3s ease;
}

.prediction-description {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
  line-height: 1.5;
}

.recommendations-card {
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  padding: 24px;
  border-radius: 12px;
  margin-bottom: 32px;
}

.recommendations-title {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #0c4a6e;
}

.recommendations-list {
  list-style: none;
  padding: 0;
}

.recommendations-list li {
  padding: 8px 0;
  border-bottom: 1px solid #bae6fd;
  color: #0c4a6e;
}

.recommendations-list li:last-child {
  border-bottom: none;
}

.disclaimer {
  background: #f5f3ff;
  border: 1px solid #e0e7ff;
  padding: 24px;
  border-radius: 12px;
}

.disclaimer-title {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #5b21b6;
}

.disclaimer-text {
  color: #374151;
  line-height: 1.6;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 300px;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #f3f3f3;
  border-top: 5px solid #7c3aed;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-container {
  text-align: center;
  background: #fee2e2;
  border: 1px solid #fecaca;
  padding: 32px;
  border-radius: 12px;
  margin: 0 auto;
  max-width: 500px;
}

.error-container h2 {
  color: #b91c1c;
  margin-bottom: 16px;
}

.error-container p {
  color: #7f1d1d;
  margin-bottom: 24px;
}

.error-actions {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
}

.btn-primary {
  display: inline-block;
  background: #7c3aed;
  color: white;
  border: none;
  padding: 12px 32px;
  border-radius: 50px;
  font-size: 18px;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.2s;
  text-decoration: none;
}

.btn-primary:hover {
  background: #6d28d9;
}

@media (max-width: 768px) {
  .results-grid {
    grid-template-columns: 1fr;
  }
  
  .action-buttons {
    flex-direction: column;
    align-items: center;
  }
  
  .summary-content {
    grid-template-columns: 1fr;
  }
  
  .error-actions {
    flex-direction: column;
    align-items: center;
  }
}
</style>
