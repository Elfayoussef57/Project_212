import { createApp } from "vue"
import App from "./App.vue"
import router from "./router"
import axios from "./plugins/axios"
import "./assets/css/main.css"

const app = createApp(App)

// Make axios available globally
app.config.globalProperties.$http = axios
app.provide("$http", axios)

app.use(router)
app.mount("#app")
