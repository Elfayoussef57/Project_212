<template>
  <div class="predictions-card">
    <h2 class="predictions-title">Analysis Predictions</h2>
    <div 
      v-for="(prediction, index) in predictions" 
      :key="index" 
      class="prediction-item"
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
    </div>
  </div>
</template>

<script>
export default {
  name: 'PredictionsCard',
  props: {
    predictions: {
      type: Array,
      required: true
    }
  }
}
</script>

<style scoped>
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
}

.prediction-item {
  margin-bottom: 24px;
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
}

.prediction-percentage {
  font-size: 18px;
  font-weight: bold;
}

.progress-bar {
  width: 100%;
  height: 16px;
  background: #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #7c3aed;
  border-radius: 8px;
  transition: width 0.3s ease;
}
</style>