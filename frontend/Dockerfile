FROM node:20-alpine as build
WORKDIR /app

# Allow injecting API URL at build time for production builds
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
ENV REACT_APP_API_URL=${VITE_API_URL}

COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps --no-audit --no-fund
COPY . .
# Build the frontend (Vite will pick up the env var during build)
RUN npm run build

FROM nginx:alpine
# Copy a minimal SPA-friendly nginx config (overrides default)
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Copy built static files into nginx html root
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
STOPSIGNAL SIGTERM
CMD ["nginx", "-g", "daemon off;"]
