<template>
  <div class="page home-page">
    <div class="container">
      <h1 class="main-title">Bienvenue à RAD-EYE</h1>
      <p class="subtitle">Téléversez des images médicales pour une analyse instantanée assistée par l’IA et une aide au diagnostic.</p>

      <div class="features-grid">
        <FeatureCard
          v-for="feature in features"
          :key="feature.id"
          :icon="feature.icon"
          :icon-color="feature.iconColor"
          :title="feature.title"
          :description="feature.description"
        />
      </div>

      <div class="cta-section">
        <router-link to="/upload" class="btn-primary">
          Démarrer
        </router-link>
      </div>
    </div>
  </div>
</template>

<script>
import FeatureCard from '../components/FeatureCard.vue'

export default {
  name: 'HomePage',
  components: {
    FeatureCard
  },
  data() {
    return {
    features: [
      {
        id: 1,
        icon: 'upload',
        iconColor: 'purple',
        title: 'Téléversement Facile',
        description: 'Téléversez en toute sécurité des images médicales dans tous les formats courants'
      },
      {
        id: 2,
        icon: 'brain',
        iconColor: 'cyan',
        title: 'Analyse par IA',
        description: 'Des réseaux neuronaux avancés analysent vos images'
      },
      {
        id: 3,
        icon: 'document',
        iconColor: 'pink',
        title: 'Rapports Détaillés',
        description: 'Générez des rapports complets pour les patients'
      }
    ]

    }
  }
}
</script>

<style scoped>
.main-title {
  font-size: 48px;
  font-weight: bold;
  text-align: center;
  color: #7c3aed;
  margin-bottom: 24px;
}

.subtitle {
  font-size: 20px;
  text-align: center;
  color: #6b7280;
  margin-bottom: 64px;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 32px;
  margin-bottom: 64px;
}

.cta-section {
  text-align: center;
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
  .main-title {
    font-size: 36px;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
}
</style>