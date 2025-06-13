<template>
  <div class="page not-found-page">
    <div class="container">
      <div class="not-found-content">
        <h1 class="not-found-title">404</h1>
        <h2 class="not-found-subtitle">Page Not Found</h2>
        <p class="not-found-message">
          The page you are looking for doesn't exist or has been moved.
        </p>
        <router-link to="/" class="btn-primary">
          Return to Home
        </router-link>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NotFoundPage'
}
</script>

<style scoped>
.not-found-page {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
}

.not-found-content {
  text-align: center;
  max-width: 500px;
  margin: 0 auto;
}

.not-found-title {
  font-size: 120px;
  font-weight: 900;
  color: #7c3aed;
  line-height: 1;
  margin-bottom: 16px;
}

.not-found-subtitle {
  font-size: 32px;
  font-weight: 700;
  color: #4b5563;
  margin-bottom: 16px;
}

.not-found-message {
  font-size: 18px;
  color: #6b7280;
  margin-bottom: 32px;
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
</style>