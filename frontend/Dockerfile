FROM node:20-alpine as build
WORKDIR /app

# Allow injecting API URL at build time for production builds
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}
ENV REACT_APP_API_URL=${VITE_API_URL}

COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps --no-audit --no-fund
COPY . .
# Build the frontend (Vite/CRA will pick up the env var during build)
RUN npm run build

FROM nginx:alpine
# Copy built static files into nginx html root
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
