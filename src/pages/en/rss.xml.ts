import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { entryHref, getPostsByLang } from '../../i18n/blog';

export async function GET(context: APIContext) {
	const posts = await getPostsByLang('en');

	return rss({
		title: 'Sabohat — dev journal',
		description: 'A dev journal on the brain-inspired SNN-based AI model.',
		site: context.site ?? 'https://ai.solvie.me',
		items: posts.map((post) => ({
			title: post.data.title,
			pubDate: post.data.date,
			description: post.data.excerpt,
			link: `${entryHref(post)}/`,
		})),
	});
}
