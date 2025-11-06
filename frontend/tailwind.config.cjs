module.exports = {
  // Ensure Tailwind scans the CRA public folder and all source files.
  // This ensures classes used in `public/index.html` and `src/**` are
  // discovered during the build and not purged.
  content: ["./public/index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [],
}
