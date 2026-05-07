import { getCollection, type CollectionEntry } from 'astro:content';
import { defaultLang, type Lang, languages } from './ui';

export type BlogEntry = CollectionEntry<'blog'>;

export function parseEntryId(id: string): { slug: string; lang: Lang } {
	const m = id.match(/^(.*)\/([a-z]{2})$/);
	if (m && m[2] in languages) {
		return { slug: m[1], lang: m[2] as Lang };
	}
	return { slug: id, lang: defaultLang };
}

export function getEntryLang(entry: BlogEntry): Lang {
	return parseEntryId(entry.id).lang;
}

export function getEntrySlug(entry: BlogEntry): string {
	return parseEntryId(entry.id).slug;
}

export function entryHref(entry: BlogEntry): string {
	const { slug, lang } = parseEntryId(entry.id);
	return lang === defaultLang ? `/blog/${slug}` : `/${lang}/blog/${slug}`;
}

export async function getPostsByLang(lang: Lang): Promise<BlogEntry[]> {
	const all = await getCollection('blog', ({ data }) => !data.draft);
	const filtered = all.filter((p) => getEntryLang(p) === lang);
	return filtered.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
