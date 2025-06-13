<template>
  <footer class="footer">
    <div class="container">
      <p>&copy; {{ currentYear }} RAD-EYE Code212. All rights reserved.</p>
    </div>
  </footer>
</template>

<script>
export default {
  name: 'AppFooter',
  computed: {
    currentYear() {
      return new Date().getFullYear()
    }
  }
}
</script>

<style scoped>
.footer {
  border-top: 1px solid #e5e7eb;
  padding: 24px 0;
  margin-top: 48px;
}

.footer .container {
  text-align: center;
  color: #6b7280;
}
</style>