import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../views/HomePage.vue'
import UploadPage from '../views/UploadPage.vue'
import ResultsPage from '../views/ResultsPage.vue'
import AboutPage from '../views/AboutPage.vue'
import NotFoundPage from '../views/NotFoundPage.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomePage,
    meta: {
      title: 'MediScan - Home'
    }
  },
  {
    path: '/upload',
    name: 'upload',
    component: UploadPage,
    meta: {
      title: 'Upload Medical Image'
    }
  },
  {
    path: '/results/:id',
    name: 'results',
    component: ResultsPage,
    props: true,
    meta: {
      title: 'Analysis Results'
    }
  },
  {
    path: '/about',
    name: 'about',
    component: AboutPage,
    meta: {
      title: 'About MediScan'
    }
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: NotFoundPage,
    meta: {
      title: 'Page Not Found'
    }
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
  scrollBehavior() {
    // Always scroll to top
    return { top: 0 }
  }
})

// Update page title based on route meta
router.beforeEach((to, from, next) => {
  document.title = to.meta.title || 'MediScan'
  next()
})

export default router