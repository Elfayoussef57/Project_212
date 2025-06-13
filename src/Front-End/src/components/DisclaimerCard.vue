<template>
  <div class="disclaimer">
    <h3 class="disclaimer-title">Important Note</h3>
    <p class="disclaimer-text">
      This analysis is provided as a diagnostic aid only and should not replace professional medical advice. 
      Please consult with a healthcare professional for proper diagnosis and treatment.
    </p>
  </div>
</template>

<script>
export default {
  name: 'DisclaimerCard'
}
</script>

<style scoped>
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
}

.disclaimer-text {
  color: #374151;
  line-height: 1.6;
}
</style>