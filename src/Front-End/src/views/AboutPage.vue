<template>
  <div class="page about-page">
    <div class="container">
      <h1 class="page-title">About MediScan</h1>
      
      <div class="about-content">
        <div class="about-section">
          <h2>Our Mission</h2>
          <p>
            MediScan is dedicated to improving healthcare outcomes through advanced AI-powered 
            medical image analysis. Our mission is to provide healthcare professionals with 
            cutting-edge tools that enhance diagnostic accuracy and efficiency.
          </p>
        </div>
        
        <div class="about-section">
          <h2>Technology</h2>
          <p>
            Our platform utilizes state-of-the-art deep learning algorithms trained on vast 
            datasets of medical images. These neural networks can detect patterns and anomalies 
            that might be missed by the human eye, providing valuable second opinions and 
            diagnostic assistance.
          </p>
        </div>
        
        <div class="about-section">
          <h2>Privacy & Security</h2>
          <p>
            We take data privacy and security seriously. All uploaded images are encrypted and 
            processed in compliance with healthcare data protection regulations. Your medical 
            data is never shared with third parties without explicit consent.
          </p>
        </div>
        
        <div class="about-section">
          <h2>Contact Us</h2>
          <p>
            For inquiries, support, or partnership opportunities, please contact us at:
            <br>
            <a href="mailto:info@mediscan.example.com" class="contact-link">info@mediscan.example.com</a>
          </p>
        </div>
      </div>
      
      <div class="cta-section">
        <router-link to="/upload" class="btn-primary">Try MediScan Now</router-link>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AboutPage'
}
</script>

<style scoped>
.page-title {
  font-size: 36px;
  font-weight: bold;
  text-align: center;
  color: #7c3aed;
  margin-bottom: 48px;
}

.about-content {
  max-width: 800px;
  margin: 0 auto 48px;
}

.about-section {
  margin-bottom: 32px;
}

.about-section h2 {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #4b5563;
}

.about-section p {
  font-size: 16px;
  line-height: 1.7;
  color: #4b5563;
}

.contact-link {
  color: #7c3aed;
  text-decoration: none;
}

.contact-link:hover {
  text-decoration: underline;
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
</style>