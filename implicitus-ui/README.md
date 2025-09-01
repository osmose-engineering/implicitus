# React + TypeScript + Vite


This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

## Getting Started

This UI requires **Node.js 18.18 or newer**. To launch the development server:

```bash
npm install
npm run dev
```

`npm run dev` starts the Vite server. Running `npm install dev` will try to install a package named `dev` which depends on the Linux-only `inotify` module and fails on macOS.

## Running Tests

Use [Vitest](https://vitest.dev) to execute the UI unit tests:

```bash
npm test
# or
npx vitest
```

## Current Progress

Our application now includes a full-featured Voronoi lattice generation backend with the following capabilities:

- **Seed-point sampling**: uniform, Poisson-disk with `min_dist`, spatially-varying `density_field`, and anisotropic sampling (`scale_field`).
- **Adaptive grid resolution**: octree subdivision driven by curvature or custom error metrics; multi-resolution support.
- **Voronoi cell construction**: surface-only and full-volume lattices, with customizable wall thickness and lattice parameters.
- **CSG operations**: smooth union, intersection, and difference with user-provided SDFs and blend radii.
- **Hybrid surface-to-solid shells**: blend curves and shell offsets to transition from thin shells to solid infill.
- **Test coverage**: comprehensive unit tests covering all core functionality.

See `BACKLOG.md` for upcoming features and enhancements.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default tseslint.config({
  extends: [
    // Remove ...tseslint.configs.recommended and replace with this
    ...tseslint.configs.recommendedTypeChecked,
    // Alternatively, use this for stricter rules
    ...tseslint.configs.strictTypeChecked,
    // Optionally, add this for stylistic rules
    ...tseslint.configs.stylisticTypeChecked,
  ],
  languageOptions: {
    // other options...
    parserOptions: {
      project: ['./tsconfig.node.json', './tsconfig.app.json'],
      tsconfigRootDir: import.meta.dirname,
    },
  },
})
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config({
  plugins: {
    // Add the react-x and react-dom plugins
    'react-x': reactX,
    'react-dom': reactDom,
  },
  rules: {
    // other rules...
    // Enable its recommended typescript rules
    ...reactX.configs['recommended-typescript'].rules,
    ...reactDom.configs.recommended.rules,
  },
})
```

