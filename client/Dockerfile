FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./
RUN npm install

COPY . .

# Expose TanStack Start's default dev port
EXPOSE 8080

CMD ["npm", "run", "dev"]