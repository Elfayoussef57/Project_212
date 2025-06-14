<template>
  <div class="page upload-page">
    <div class="container">
      <h1 class="page-title">Entrer Une Image Médicale</h1>
      <p class="page-subtitle">Veuillez téléverser une image médicale claire et en haute résolution pour l’analyse.</p>

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
          <h3 class="upload-title">Glissez-déposez votre image ici ou cliquez pour parcourir vos fichiers.</h3>
          <p class="upload-subtitle">Prend en charge les formats JPG, PNG, DICOM et autres formats d’images médicales.</p>
        </div>

        <div v-if="selectedFile" class="file-preview">
          <div class="file-info">
            <div class="success-message">Fichier Sélectionner:</div>
            <div class="file-name">{{ selectedFile.name }}</div>
            <img v-if="previewUrl" :src="previewUrl" alt="Preview" class="preview-image" />
            <button class="remove-file-btn" @click.stop="removeFile">Supprimer le Fichier</button>
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

      <!-- Always show the analyze button, but disable it when no file is selected -->
      <div class="analyze-section">
        <button 
          class="btn-analyze" 
          @click="analyzeImage" 
          :disabled="!selectedFile || isAnalyzing"
        >
          {{ isAnalyzing ? 'Analyse en cours...' : 'Analyser le scan' }}
        </button>
      </div>

      <!-- Results Section -->
      <div v-if="analysisResult" class="results-section">
        <h2 class="results-title">Résultats de l'analyse</h2>
        
        <div class="result-summary">
          <div class="result-badge" :class="getResultClass(analysisResult.result)">
            {{ analysisResult.result }}
          </div>
          <div class="confidence-meter">
            <span class="confidence-label">Confiance:</span>
            <div class="confidence-bar">
              <div 
                class="confidence-fill" 
                :style="{ width: (analysisResult.confidence * 100) + '%' }"
                :class="getConfidenceClass(analysisResult.confidence)"
              ></div>
            </div>
            <span class="confidence-value">{{ (analysisResult.confidence * 100).toFixed(1) }}%</span>
          </div>
        </div>

        <div class="images-grid" v-if="analysisResult.heatmap_url!= 'http://127.0.0.1:5000undefined'">
          <div class="result-image-card">
            <h3>Image originale</h3>
            <div class="image-container">
              <img :src="analysisResult.image_url" alt="Image originale" class="result-image" />
            </div>
          </div>
          
          <div class="result-image-card">
            <h3>Carte de chaleur</h3>
            <div class="image-container">
              <img :src="analysisResult.heatmap_url" alt="Carte de chaleur" class="result-image" />
            </div>
          </div>
        </div>

        <div class="analysis-explanation" v-if="analysisResult.heatmap_url!= 'http://127.0.0.1:5000undefined'">
          <h3>Interprétation</h3>
          <p v-if="isPneumonia(analysisResult.result)">
            L'analyse indique la présence probable d'une pneumonie avec un niveau de confiance de 
            {{ (analysisResult.confidence * 100).toFixed(1) }}%. La carte de chaleur met en évidence 
            les zones des poumons qui présentent des opacités caractéristiques d'une infection pulmonaire.
          </p>
          <p v-else>
            L'analyse n'a pas détecté de signes de pneumonie sur cette radiographie. 
            Le niveau de confiance de cette prédiction est de {{ (analysisResult.confidence * 100).toFixed(1) }}%.
          </p>
          <div class="disclaimer">
            <strong>Note importante:</strong> Ce résultat est fourni à titre indicatif seulement et ne remplace pas 
            l'avis d'un professionnel de santé qualifié. Veuillez consulter un médecin pour une interprétation 
            définitive de cette image.
          </div>
        </div>

<div class="action-buttons" >
    <button @click="showDownloadModal = true" class="btn-download" v-if="analysisResult.heatmap_url!= 'http://127.0.0.1:5000undefined'">
      <DownloadIcon />
      Télécharger les résultats
    </button>
    <button @click="newAnalysis" class="btn-new">
      <RefreshIcon />
      Nouvelle analyse
    </button>
  </div>

  <!-- Modal pour télécharger le rapport -->
  <div v-if="showDownloadModal" class="modal-overlay">
    <div class="modal-content">
      <h3>Générer le rapport PDF</h3>
      <div class="form-group">
        <label for="firstName">Prénom</label>
        <input 
          id="firstName" 
          type="text" 
          v-model="patientFirstName" 
          placeholder="Entrez le prénom"
          class="form-input"
        />
        </div>
            <div class="form-group">
              <label for="lastName">Nom</label>
              <input 
                id="lastName" 
                type="text" 
                v-model="patientLastName" 
                placeholder="Entrez le nom"
                class="form-input"
              />
            </div>
            <div class="modal-actions">
              <button @click="showDownloadModal = false" class="btn-cancel">
                Annuler
              </button>
              <button @click="generatePDFReport" class="btn-generate" :disabled="!patientFirstName || !patientLastName">
                Générer le rapport
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Error display -->
      <div v-if="error" class="error-message">
        <AlertIcon />
        <p>{{ error }}</p>
        <button @click="error = null" class="btn-close">Fermer</button>
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
import apiClient from '../plugins/axios'
import UploadIcon from '../components/icons/UploadIcon.vue'
import DownloadIcon from '../components/icons/DownloadIcon.vue'
import RefreshIcon from '../components/icons/RefreshIcon.vue'
import AlertIcon from '../components/icons/AlertIcon.vue'
import FormatTags from '../components/FormatTags.vue'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

export default {
  name: 'UploadPage',
  components: {
    UploadIcon,
    DownloadIcon,
    RefreshIcon,
    AlertIcon,
    FormatTags
  },
  data() {
    return {
      selectedFile: null,
      previewUrl: null,
      isDragging: false,
      isAnalyzing: false,
      uploadProgress: 0,
      analysisResult: null,
      error: null,
      showDownloadModal: false,
      patientFirstName: '',
      patientLastName: ''
    }
  },
  methods: {
    handleDragOver(e) {
      e.preventDefault()
      this.isDragging = true
    },
    handleDragLeave() {
      this.isDragging = false
    },
    handleDrop(e) {
      e.preventDefault()
      this.isDragging = false
      
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this.setSelectedFile(e.dataTransfer.files[0])
      }
    },
    handleFileSelect(e) {
      if (e.target.files && e.target.files.length > 0) {
        this.setSelectedFile(e.target.files[0])
      }
    },
    setSelectedFile(file) {
      this.selectedFile = file
      this.analysisResult = null // Clear previous results
      this.error = null // Clear previous errors
      
      if (file && file.type.startsWith('image/')) {
        this.previewUrl = URL.createObjectURL(file)
      }
    },
    removeFile() {
      this.selectedFile = null
      this.previewUrl = null
      this.uploadProgress = 0
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = ''
      }
    },
    async analyzeImage() {
      if (!this.selectedFile) return
      
      this.isAnalyzing = true
      this.uploadProgress = 0
      this.error = null
      
      try {
        // Create FormData to send the file
        const formData = new FormData()
        formData.append('file', this.selectedFile) // Note: API expects 'file', not 'image'
        
        // Configure axios request with progress tracking
        const config = {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 300000, // 5 minutes timeout
          onUploadProgress: (progressEvent) => {
            if (progressEvent.lengthComputable) {
              this.uploadProgress = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              )
            }
          }
        }
        
        // Call the backend API using the configured apiClient
        const response = await apiClient.post('/api/analyze-scan', formData, config)
        
        // Store the analysis result
        this.analysisResult = response.data
        const BASE_URL = 'http://127.0.0.1:5000'  // ou localhost si tu préfères

        this.analysisResult.image_url = BASE_URL + this.analysisResult.image_url
        this.analysisResult.resized_image_url = BASE_URL + this.analysisResult.resized_image_url
        this.analysisResult.heatmap_url = BASE_URL + this.analysisResult.heatmap_url

        console.log(response.data, 'Analysis result received')
        // Scroll to results
        this.$nextTick(() => {
          const resultsSection = document.querySelector('.results-section')
          if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth' })
          }
        })
        
      } catch (error) {
        console.error('Error analyzing image:', error)
        
        let errorMessage = 'Une erreur est survenue lors de l\'analyse. Veuillez réessayer.'
        
        if (error.response) {
          // Server responded with error status
          if (error.response.data && error.response.data.error) {
            errorMessage = error.response.data.error
            if (error.response.data.message) {
              errorMessage += ': ' + error.response.data.message
            }
          } else {
            switch (error.response.status) {
              case 400:
                errorMessage = 'Format de fichier non supporté ou fichier corrompu.'
                break
              case 413:
                errorMessage = 'Le fichier est trop volumineux.'
                break
              case 500:
                errorMessage = 'Erreur serveur. Veuillez réessayer plus tard.'
                break
              default:
                errorMessage = `Erreur ${error.response.status}: Erreur inconnue`
            }
          }
        } else if (error.request) {
          // Network error
          errorMessage = 'Erreur de connexion. Vérifiez votre connexion internet.'
        } else if (error.code === 'ECONNABORTED') {
          // Timeout error
          errorMessage = 'L\'analyse prend trop de temps. Veuillez réessayer.'
        }
        
        this.error = errorMessage
      } finally {
        this.isAnalyzing = false
        this.uploadProgress = 0
      }
    },
    isPneumonia(result) {
      return result && result.toLowerCase().includes('pneumonie')
    },
    getResultClass(result) {
      if (!result) return ''
      
      if (this.isPneumonia(result)) {
        return 'result-positive'
      } else if (result.toLowerCase().includes('non valide')) {
        return 'result-invalid'
      } else {
        return 'result-negative'
      }
    },
    getConfidenceClass(confidence) {
      if (confidence > 0.8) return 'confidence-high'
      if (confidence > 0.6) return 'confidence-medium'
      return 'confidence-low'
    },

async generatePDFReport()
{
  if (!this.patientFirstName || !this.patientLastName) {
    this.error = "Veuillez entrer le nom et prénom du patient"
    return
  }

  try {
    this.isAnalyzing = true

    // Créer un nouveau document PDF
    const doc = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    })

    // Ajouter un fond de page stylisé
    doc.setFillColor(240, 248, 255) // Bleu très clair
    doc.rect(0, 0, 210, 297, "F")

    // Ajouter une bordure en haut
    doc.setDrawColor(70, 130, 180) // Bleu acier
    doc.setLineWidth(0.5)
    doc.line(10, 10, 200, 10)

    // En-tête avec style
    doc.setFillColor(70, 130, 180) // Bleu acier
    doc.rect(10, 15, 190, 15, "F")

    doc.setFont("helvetica", "bold")
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(18)
    doc.text("RAPPORT D'ANALYSE MÉDICALE", 105, 25, { align: "center" })

    // Informations du patient dans un cadre
    doc.setDrawColor(100, 100, 100)
    doc.setLineWidth(0.3)
    doc.roundedRect(10, 35, 190, 25, 3, 3, "S")

    doc.setFont("helvetica", "bold")
    doc.setTextColor(70, 130, 180)
    doc.setFontSize(12)
    doc.text("INFORMATIONS PATIENT", 15, 43)

    doc.setFont("helvetica", "normal")
    doc.setTextColor(60, 60, 60)
    doc.setFontSize(11)
    doc.text(`Nom et prénom: ${this.patientFirstName} ${this.patientLastName}`, 15, 50)
    doc.text(`Date d'examen: ${new Date().toLocaleDateString("fr-FR")}`, 15, 55)

    // Numéro de dossier fictif
    const dossierNum = Math.floor(10000 + Math.random() * 90000)
    doc.text(`N° de dossier: ${dossierNum}`, 130, 50)

    // Résultats dans un cadre
    doc.setDrawColor(100, 100, 100)
    doc.roundedRect(10, 65, 190, 25, 3, 3, "S")

    doc.setFont("helvetica", "bold")
    doc.setTextColor(70, 130, 180)
    doc.text("RÉSULTATS D'ANALYSE", 15, 73)

    // Diagnostic avec couleur conditionnelle
    doc.setFont("helvetica", "bold")
    doc.setTextColor(this.isPneumonia(this.analysisResult.result) ? "#b91c1c" : "#065f46")
    doc.text(`Diagnostic: ${this.analysisResult.result}`, 15, 80)

    // Niveau de confiance avec barre visuelle
    doc.setFont("helvetica", "normal")
    doc.setTextColor(60, 60, 60)
    doc.text(`Confiance: ${(this.analysisResult.confidence * 100).toFixed(1)}%`, 15, 87)

    // Barre de confiance visuelle
    const confidenceWidth = this.analysisResult.confidence * 100
    doc.setFillColor(this.isPneumonia(this.analysisResult.result) ? "#fecaca" : "#d1fae5")
    doc.rect(80, 84, confidenceWidth, 4, "F")
    doc.setDrawColor(100, 100, 100)
    doc.rect(80, 84, 100, 4, "S")

    // Section pour les images avec fond gris clair
    const imagesY = 95
    doc.setFillColor(245, 245, 245) // Gris très clair
    doc.rect(10, imagesY, 95, 110, "F") // Fond gauche
    doc.rect(105, imagesY, 95, 110, "F") // Fond droite

    // Titres des images
    doc.setFont("helvetica", "normal")
    doc.setTextColor(80, 80, 80)
    doc.setFontSize(10)
    doc.text("Image originale", 57.5, imagesY + 8, { align: "center" })
    doc.text("Carte de chaleur", 152.5, imagesY + 8, { align: "center" })

    // Méthode alternative pour ajouter les images directement
    const addImageDirectly = async (imageUrl, x, y, width, height) => {
      return new Promise((resolve, reject) => {
        try {
          const img = new Image()
          img.crossOrigin = "Anonymous" // Important pour CORS

          img.onload = () => {
            // Créer un canvas pour manipuler l'image
            const canvas = document.createElement("canvas")
            canvas.width = img.width
            canvas.height = img.height
            const ctx = canvas.getContext("2d")
            ctx.drawImage(img, 0, 0)

            try {
              // Convertir en base64 et ajouter au PDF
              const imgData = canvas.toDataURL("image/jpeg")
              doc.addImage(imgData, "JPEG", x, y, width, height)
              resolve()
            } catch (e) {
              console.error("Erreur lors de l'ajout de l'image au PDF:", e)
              reject(e)
            }
          }

          img.onerror = (e) => {
            console.error("Erreur de chargement de l'image:", e)
            reject(e)
          }

          // Déclencher le chargement de l'image
          img.src = imageUrl
        } catch (error) {
          console.error("Erreur générale:", error)
          reject(error)
        }
      })
    }

    // Ajouter les images
    try {
      // Image originale
      await addImageDirectly(this.analysisResult.image_url, 15, imagesY + 15, 85, 85)

      // Carte de chaleur
      if (this.analysisResult.heatmap_url && this.analysisResult.heatmap_url !== "http://127.0.0.1:5000undefined") {
        await addImageDirectly(this.analysisResult.heatmap_url, 110, imagesY + 15, 85, 85)
      }
    } catch (error) {
      console.error("Erreur lors de l'ajout des images:", error)
      // Ajouter un texte d'erreur dans le PDF
      doc.setTextColor(255, 0, 0)
      doc.setFontSize(10)
      doc.text("Impossible de charger les images", 105, imagesY + 60, { align: "center" })
    }

    // Mettre à jour la position Y pour la section suivante
    const yPosition = imagesY + 120

    // Ajouter un champ à remplir
    doc.setDrawColor(100, 100, 100)
    doc.roundedRect(10, yPosition, 190, 40, 3, 3, "S")

    doc.setFont("helvetica", "bold")
    doc.setTextColor(70, 130, 180)
    doc.text("OBSERVATIONS DU MÉDECIN", 15, yPosition + 7)

    doc.setFont("helvetica", "italic")
    doc.setTextColor(150, 150, 150)
    doc.text("(Espace réservé aux commentaires du praticien)", 15, yPosition + 15)

    // Lignes pour écrire
    doc.setDrawColor(200, 200, 200)
    doc.setLineWidth(0.2)
    for (let i = 0; i < 3; i++) {
      doc.line(15, yPosition + 20 + i * 7, 195, yPosition + 20 + i * 7)
    }

    // Ajouter un champ pour la signature
    //doc.setDrawColor(100, 100, 100)
    //doc.setLineWidth(0.3)
    //doc.roundedRect(130, yPosition + 5, 65, 30, 3, 3, "S")

    //doc.setFont("helvetica", "italic")
    //doc.setTextColor(150, 150, 150)
    //doc.setFontSize(8)
    //doc.text("Signature et cachet", 162.5, yPosition + 32, { align: "center" })"

    // Pied de page avec disclaimer
    const footerY = 285
    doc.setDrawColor(70, 130, 180)
    doc.setLineWidth(0.5)
    doc.line(10, footerY - 5, 200, footerY - 5)

    doc.setFont("helvetica", "italic")
    doc.setFontSize(8)
    doc.setTextColor(100, 100, 100)
    doc.text(
      "Ce résultat est fourni à titre indicatif seulement et ne remplace pas l'avis d'un professionnel de santé qualifié.",
      105,
      footerY,
      { align: "center", maxWidth: 180 },
    )

    // Numéro de page
    doc.text(
      `Page 1/1 - Généré le ${new Date().toLocaleDateString("fr-FR")} à ${new Date().toLocaleTimeString("fr-FR")}`,
      105,
      footerY + 5,
      { align: "center" },
    )

    // Sauvegarder le PDF
    doc.save(`rapport_${this.patientLastName}_${this.patientFirstName}.pdf`)

    this.showDownloadModal = false
  } catch (error) {
    console.error("Erreur lors de la génération du PDF:", error)
    this.error = "Une erreur est survenue lors de la génération du rapport PDF"
  } finally {
    this.isAnalyzing = false
  }
},
    newAnalysis() {
      this.removeFile()
      this.analysisResult = null
      this.error = null
      
      // Scroll to top
      window.scrollTo({ top: 0, behavior: 'smooth' })
    },
    beforeUnmount() {
      if (this.previewUrl) {
        URL.revokeObjectURL(this.previewUrl)
      }
    }
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
  display: block;           /* ou inline-block */
  text-align: center;
  margin: 0 auto;           /* pour le centrer horizontalement dans son conteneur */
}


.analyze-section {
  text-align: center;
  margin-top: 32px;
  margin-bottom: 32px;
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
  opacity: 0.7;
}

/* Results Section Styles */
.results-section {
  margin-top: 60px;
  padding: 32px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.results-title {
  font-size: 28px;
  font-weight: bold;
  color: #374151;
  margin-bottom: 24px;
  text-align: center;
}

.result-summary {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 32px;
}

.result-badge {
  padding: 12px 24px;
  border-radius: 50px;
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 16px;
}

.result-positive {
  background-color: #fee2e2;
  color: #b91c1c;
}

.result-negative {
  background-color: #d1fae5;
  color: #065f46;
}

.result-invalid {
  background-color: #fef3c7;
  color: #92400e;
}

.confidence-meter {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  max-width: 500px;
}

.confidence-label {
  font-weight: 600;
  color: #4b5563;
  min-width: 80px;
}

.confidence-bar {
  flex-grow: 1;
  height: 12px;
  background: #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
}

.confidence-fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.5s ease;
}

.confidence-high {
  background-color: #10b981;
}

.confidence-medium {
  background-color: #f59e0b;
}

.confidence-low {
  background-color: #ef4444;
}

.confidence-value {
  font-weight: 600;
  color: #4b5563;
  min-width: 60px;
  text-align: right;
}

.images-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  margin-bottom: 32px;
}

.result-image-card {
  background: #f9fafb;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.result-image-card h3 {
  padding: 16px;
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #374151;
  background: #f3f4f6;
  text-align: center;
}

.image-container {
  padding: 16px;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 250px;
}

.result-image {
  max-width: 100%;
  max-height: 250px;
  object-fit: contain;
}

.analysis-explanation {
  background: #f8fafc;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 32px;
}

.analysis-explanation h3 {
  font-size: 20px;
  font-weight: 600;
  color: #374151;
  margin-top: 0;
  margin-bottom: 16px;
}

.analysis-explanation p {
  color: #4b5563;
  line-height: 1.6;
  margin-bottom: 16px;
}

.disclaimer {
  background: #fffbeb;
  border-left: 4px solid #f59e0b;
  padding: 16px;
  color: #92400e;
  font-size: 14px;
  line-height: 1.5;
}

.action-buttons {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-top: 32px;
}

.btn-download,
.btn-new {
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

.btn-download:hover {
  background: #047857;
}

.btn-new {
  background: #3b82f6;
  color: white;
}

.btn-new:hover {
  background: #2563eb;
}

.error-message {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #fee2e2;
  border-left: 4px solid #ef4444;
  padding: 16px;
  margin: 32px 0;
  color: #b91c1c;
  position: relative;
}

.btn-close {
  position: absolute;
  top: 8px;
  right: 8px;
  background: none;
  border: none;
  color: #b91c1c;
  cursor: pointer;
  font-size: 12px;
  padding: 4px 8px;
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
  
  .images-grid {
    grid-template-columns: 1fr;
  }
  
  .action-buttons {
    flex-direction: column;
    align-items: center;
  }
}

/* Styles pour la modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  padding: 32px;
  border-radius: 12px;
  width: 100%;
  max-width: 500px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.modal-content h3 {
  font-size: 24px;
  margin-top: 0;
  margin-bottom: 24px;
  text-align: center;
  color: #7c3aed;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #374151;
}

.form-input {
  width: 100%;
  padding: 12px 16px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 16px;
  transition: border-color 0.2s;
}

.form-input:focus {
  outline: none;
  border-color: #7c3aed;
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 24px;
}

.btn-cancel {
  padding: 12px 24px;
  background: #f3f4f6;
  color: #374151;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-cancel:hover {
  background: #e5e7eb;
}

.btn-generate {
  padding: 12px 24px;
  background: #7c3aed;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-generate:hover:not(:disabled) {
  background: #6d28d9;
}

.btn-generate:disabled {
  background: #c4b5fd;
  cursor: not-allowed;
  opacity: 0.7;
}
</style>