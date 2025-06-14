<template>
  <div class="page about-page">
    <div class="container">
      <h1 class="page-title">À propos RAD-EYE</h1>
      
      <div class="images-section">
        <div class="image-container">
          <img src="../assets/logo.jpg" alt="Logo RAD-EYE" class="about-image" />
        </div>
        <div class="image-container">
          <img src="../assets/code212.jpg" alt="Code212" class="about-image" />
        </div>
      </div>
      
      <div class="description-section">
        <div class="description-content">
          <p class="description-text">
            Notre projet, <span class="highlight">Rad-Eye</span>, est un système de détection de la pneumonie basé sur l'intelligence artificielle. Il s'agit d'une initiative innovante développée par une équipe d'étudiants passionnés, ayant pour objectif de contribuer au diagnostic précoce de la pneumonie grâce aux technologies de l'IA. En tirant parti des techniques d'apprentissage profond, notamment les réseaux de neurones convolutifs (CNN), notre modèle est entraîné à analyser des radiographies thoraciques afin d'identifier avec une grande précision les cas de pneumonie.
          </p>
          
          <p class="description-text">
            Ce projet se veut un pont entre la technologie et la santé, en proposant un outil de diagnostic rapide, fiable et accessible, destiné à soutenir les professionnels du domaine médical. Réalisé sous la supervision de <span class="highlight">M. Berrada</span> à l'<span class="highlight">ENSA de Tanger</span>, il reflète notre engagement à mettre l'intelligence artificielle au service du bien commun, en améliorant l'accessibilité aux soins et en contribuant à l'évolution des diagnostics médicaux.
          </p>
        </div>
      </div>

      <div class="cta-section">
        <router-link to="/upload" class="btn-primary">Essayer RAD-EYE maintenant</router-link>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AboutPage'
}
</script>

<style scoped>
.page-title {
  font-size: 42px;
  font-weight: bold;
  text-align: center;
  color: #7c3aed;
  margin-bottom: 60px;
  text-shadow: 0 2px 4px rgba(124, 58, 237, 0.1);
}

.images-section {
  display: flex;
  justify-content: center;
  gap: 40px;
  margin-bottom: 60px;
  flex-wrap: wrap;
}

.image-container {
  position: relative;
  overflow: hidden;
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.image-container:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
}

.about-image {
  width: 300px;
  height: 200px;
  object-fit: cover;
  border-radius: 20px;
  transition: transform 0.3s ease;
}

.about-image:hover {
  transform: scale(1.05);
}

.description-section {
  max-width: 900px;
  margin: 0 auto;
  padding: 40px;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 24px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
  position: relative;
  overflow: hidden;
}

.description-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #7c3aed, #3b82f6, #10b981);
}

.description-content {
  position: relative;
  z-index: 1;
}

.description-text {
  font-size: 18px;
  line-height: 1.8;
  color: #374151;
  margin-bottom: 24px;
  text-align: justify;
  font-weight: 400;
  letter-spacing: 0.3px;
}

.description-text:last-child {
  margin-bottom: 0;
}

.highlight {
  color: #7c3aed;
  font-weight: 600;
  background: linear-gradient(120deg, rgba(124, 58, 237, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%);
  padding: 2px 6px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.highlight:hover {
  background: linear-gradient(120deg, rgba(124, 58, 237, 0.15) 0%, rgba(124, 58, 237, 0.08) 100%);
  transform: translateY(-1px);
}

/* Decorative elements */
.description-section::after {
  content: '';
  position: absolute;
  bottom: -50px;
  right: -50px;
  width: 100px;
  height: 100px;
  background: radial-gradient(circle, rgba(124, 58, 237, 0.1) 0%, transparent 70%);
  border-radius: 50%;
}

.cta-section {
  text-align: center;
  margin-top: 48px;
}

.btn-primary {
  display: inline-block;
  background: linear-gradient(135deg, #7c3aed, #6d28d9);
  color: white;
  border: none;
  padding: 16px 40px;
  border-radius: 50px;
  font-size: 18px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  text-decoration: none;
  box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.btn-primary:hover {
  background: linear-gradient(135deg, #6d28d9, #5b21b6);
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
}

@media (max-width: 768px) {
  .page-title {
    font-size: 32px;
    margin-bottom: 40px;
  }
  
  .images-section {
    gap: 20px;
    margin-bottom: 40px;
  }
  
  .about-image {
    width: 250px;
    height: 170px;
  }
  
  .description-section {
    margin: 0 20px;
    padding: 30px 20px;
  }
  
  .description-text {
    font-size: 16px;
    line-height: 1.7;
  }
}

@media (max-width: 480px) {
  .images-section {
    flex-direction: column;
    align-items: center;
  }
  
  .about-image {
    width: 280px;
    height: 180px;
  }
  
  .description-section {
    margin: 0 10px;
    padding: 25px 15px;
  }
}
</style>