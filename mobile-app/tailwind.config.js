/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,js,ts}"],
  theme: {
    extend: {
      // Rozmiary celów dotykowych w "dp" (mapowane 1:1 na px bazowe).
      // Zakres adaptacji: 48dp (standard WCAG/Material) -> 72dp (silne drżenie).
      spacing: {
        "touch-sm": "48px",
        "touch-md": "60px",
        "touch-lg": "72px",
      },
    },
  },
  plugins: [],
};
