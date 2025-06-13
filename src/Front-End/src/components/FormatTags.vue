<template>
  <div class="format-tags">
    <span 
      v-for="format in formats" 
      :key="format" 
      class="format-tag"
    >
      {{ format }}
    </span>
  </div>
</template>

<script>
export default {
  name: 'FormatTags',
  data() {
    return {
      formats: ['JPG', 'PNG', 'DICOM']
    }
  }
}
</script>

<style scoped>
.format-tags {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-bottom: 32px;
}

.format-tag {
  background: #f3e8ff;
  color: #7c3aed;
  padding: 8px 24px;
  border-radius: 50px;
  font-weight: 600;
}
</style>