// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
	site: 'https://ai.solvie.me',
	vite: {
		plugins: [tailwindcss()],
	},
});
