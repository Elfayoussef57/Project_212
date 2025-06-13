<template>
  <header class="header">
    <div class="container">
      <router-link to="/" class="logo">RAD-EYE</router-link>
      <nav class="nav">
        <ul>
          <li>
            <router-link to="/" class="nav-link">Home</router-link>
          </li>
          <li>
            <router-link to="/upload" class="nav-link">Upload</router-link>
          </li>
          <li>
            <router-link to="/about" class="nav-link">About</router-link>
          </li>
        </ul>
      </nav>
    </div>
  </header>
</template>

<script>
export default {
  name: 'AppHeader'
}
</script>

<style scoped>
.header {
  border-bottom: 1px solid #e5e7eb;
  background: white;
}

.header .container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
}

.logo {
  font-size: 24px;
  font-weight: bold;
  color: #7c3aed;
  text-decoration: none;
}

.nav ul {
  display: flex;
  list-style: none;
  gap: 24px;
}

.nav-link {
  text-decoration: none;
  color: #374151;
  font-weight: 500;
  transition: color 0.2s;
}

.nav-link:hover,
.router-link-active {
  color: #7c3aed;
}

@media (max-width: 768px) {
  .header .container {
    flex-direction: column;
    gap: 16px;
  }
  
  .nav ul {
    gap: 16px;
  }
}
</style>