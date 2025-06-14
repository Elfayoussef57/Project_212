<template>
  <header class="header">
    <div class="container">
      <router-link to="/" class="logo">
        <img src="../assets/logo.jpg" alt="RAD-EYE Logo" class="logo-image" />
        <span class="logo-text">RAD-EYE</span>
      </router-link>
      <nav class="nav">
        <ul>
          <li>
            <router-link to="/" class="nav-link">Accueil</router-link>
          </li>
          <li>
            <router-link to="/upload" class="nav-link">Analyser</router-link>
          </li>
          <li>
            <router-link to="/about" class="nav-link">À propos</router-link>
          </li>
        </ul>
      </nav>
    </div>
  </header>
</template>

<script>
export default {
  name: 'AppHeader',
}
</script>

<style scoped>
.header {
  background-color: #f0f0f0;
  padding: 20px 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.nav ul {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
}

.nav li {
  margin-left: 20px;
}

.nav-link {
  text-decoration: none;
  color: #333;
  font-weight: 500;
  transition: color 0.3s ease;
}

.nav-link:hover {
  color: #7c3aed;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 24px;
  font-weight: bold;
  color: #7c3aed;
  text-decoration: none;
}

.logo-image {
  width: 40px;
  height: 40px;
  object-fit: contain;
  border-radius: 8px;
}

.logo-text {
  font-size: 24px;
  font-weight: bold;
  color: #7c3aed;
}
</style>