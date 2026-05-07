// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
	site: 'https://ai.solvie.me',
	i18n: {
		defaultLocale: 'uz',
		locales: ['uz', 'en'],
		routing: {
			prefixDefaultLocale: false,
		},
	},
	integrations: [
		sitemap({
			filter: (page) => !page.includes('/404'),
			changefreq: 'weekly',
			priority: 0.7,
			lastmod: new Date(),
			i18n: {
				defaultLocale: 'uz',
				locales: {
					uz: 'uz-UZ',
					en: 'en-US',
				},
			},
			serialize(item) {
				const url = item.url;
				if (url === 'https://ai.solvie.me/') {
					item.priority = 1.0;
					item.changefreq = 'daily';
				} else if (url === 'https://ai.solvie.me/blog/') {
					item.priority = 0.9;
					item.changefreq = 'daily';
				} else if (url.includes('/blog/')) {
					item.priority = 0.8;
					item.changefreq = 'monthly';
				} else if (url.includes('/about/') || url.includes('/projects/')) {
					item.priority = 0.6;
					item.changefreq = 'monthly';
				}
				return item;
			},
		}),
	],
	vite: {
		plugins: [tailwindcss()],
	},
});
