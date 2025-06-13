<template>
  <div class="feature-card">
    <div class="feature-icon" :class="iconColor">
      <component :is="iconComponent" />
    </div>
    <h2 class="feature-title">{{ title }}</h2>
    <p class="feature-description">{{ description }}</p>
  </div>
</template>

<script>
import UploadIcon from './icons/UploadIcon.vue'
import BrainIcon from './icons/BrainIcon.vue'
import DocumentIcon from './icons/DocumentIcon.vue'

export default {
  name: 'FeatureCard',
  components: {
    UploadIcon,
    BrainIcon,
    DocumentIcon
  },
  props: {
    icon: {
      type: String,
      required: true
    },
    iconColor: {
      type: String,
      required: true
    },
    title: {
      type: String,
      required: true
    },
    description: {
      type: String,
      required: true
    }
  },
  computed: {
    iconComponent() {
      const iconMap = {
        upload: 'UploadIcon',
        brain: 'BrainIcon',
        document: 'DocumentIcon'
      }
      return iconMap[this.icon] || 'UploadIcon'
    }
  }
}
</script>

<style scoped>
.feature-card {
  border-left: 4px solid #7c3aed;
  padding: 24px;
}

.feature-icon {
  width: 64px;
  height: 64px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.feature-icon.purple {
  background: #7c3aed;
  color: white;
}

.feature-icon.cyan {
  background: #06b6d4;
  color: white;
}

.feature-icon.pink {
  background: #ec4899;
  color: white;
}

.feature-title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 12px;
}

.feature-description {
  color: #6b7280;
  font-size: 16px;
}
</style>