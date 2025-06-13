<template>
  <div class="result-card">
    <h2 class="result-title">{{ title }}</h2>
    <div class="image-container">
      <img :src="imageSrc" :alt="alt" class="result-image" />
    </div>
  </div>
</template>

<script>
export default {
  name: 'ResultCard',
  props: {
    title: {
      type: String,
      required: true
    },
    imageSrc: {
      type: String,
      required: true
    },
    alt: {
      type: String,
      required: true
    }
  }
}
</script>

<style scoped>
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
}

.image-container {
  aspect-ratio: 1;
  background: #f3f4f6;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}

.result-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
</style>