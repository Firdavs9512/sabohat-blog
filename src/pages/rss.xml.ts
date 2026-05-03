import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
	const posts = (await getCollection('blog', ({ data }) => !data.draft)).sort(
		(a, b) => b.data.date.valueOf() - a.data.date.valueOf(),
	);

	return rss({
		title: 'Sabohat — dev journal',
		description:
			'SNN asosidagi miyaga o\'xshash AI model ustidagi ish kundaligi.',
		site: context.site ?? 'https://ai.solvie.me',
		items: posts.map((post) => ({
			title: post.data.title,
			pubDate: post.data.date,
			description: post.data.excerpt,
			link: `/blog/${post.id}/`,
		})),
	});
}
