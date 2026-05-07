export const languages = {
	uz: "O'zbek",
	en: 'English',
} as const;

export const defaultLang = 'uz' as const;

export type Lang = keyof typeof languages;

export const ui = {
	uz: {
		'site.title': 'Sabohat — AI tadqiqot kundaligi',
		'site.description':
			"Sabohat (Samantha) — miyaga o'xshash ishlovchi SNN AI modeli. O'zbek tili uchun, self-learning va continual learning fokusida olib borilayotgan ish kundaligi.",
		'site.keywords':
			"Sabohat, Samantha, AI, SNN, spiking neural network, sun'iy intellekt, o'zbek tili, neural network, continual learning, brain-inspired AI, machine learning",
		'site.locale': 'uz_UZ',

		'nav.home': 'bosh',
		'nav.blog': 'yozuvlar',
		'nav.projects': 'loyihalar',
		'nav.about': 'men',

		'home.intro.lead': 'asosida ishlaydigan, miyaga o\'xshash AI model. Maqsad — modelni o\'z-o\'zidan o\'rgana oladigan holatga olib chiqish va',
		'home.intro.cl': 'muammosini (catastrophic forgetting) sezilarli darajada kamaytirish.',
		'home.info.architecture': 'arxitektura',
		'home.info.architectureValue': "SNN · miyaga o'xshash · spike-based",
		'home.info.goal': 'maqsad',
		'home.info.goalValue': "o'z-o'zidan o'rganish + CL forgetting kamaytirish",
		'home.info.experiments': 'sinovlar',
		'home.info.site': 'sayt',
		'home.divider.recent': "so'nggi yozuvlar",
		'home.allPosts': '→ barcha yozuvlar',

		'blog.heading': '/yozuvlar',
		'blog.subtitle': "// Sabohat (miyaga o'xshash SNN AI) ustida olib borilayotgan ish kundaligi.",
		'blog.countSuffix': 'ta yozuv',
		'blog.back': '← yozuvlar',
		'blog.backAll': '← barcha yozuvlar',

		'about.heading': '/men-haqimda',
		'about.title': 'Men haqimda · Sabohat',
		'about.description':
			"Sabohat — miyaga o'xshash ishlovchi SNN (Spiking Neural Network) AI modeli. Loyihaning maqsadi, bosqichlari va shu yerda nima yozilishi haqida.",

		'projects.heading': '/loyihalar',
		'projects.subtitle': '// Hozir ustida ishlayotgan narsalar.',
		'projects.title': 'Loyihalar · Sabohat',
		'projects.description':
			"Hozir ustida ishlayotgan AI loyihalari: Sabohat (miyaga o'xshash SNN) va solvie.me asosiy domeni.",
		'projects.statusActive': 'faol',

		'404.title': 'Sahifa topilmadi · Sabohat',
		'404.description': "Siz qidirgan sahifa topilmadi.",
		'404.text': "Siz qidirgan sahifa mavjud emas. Balki nomi o'zgargan yoki o'chirilgan.",
		'404.home': '← bosh sahifa',
		'404.art': ' ┌────────────────────────────┐\n │  404 · sahifa topilmadi    │\n │  ─ signal lost ─           │\n └────────────────────────────┘',

		'footer.copy': '© {year} sabohat',
		'footer.built': 'built with astro + ☕',
		'langSwitch.label': 'til',
	},
	en: {
		'site.title': 'Sabohat — AI research journal',
		'site.description':
			'Sabohat (Samantha) — a brain-inspired SNN AI model. A research journal focused on the Uzbek language, self-learning and continual learning.',
		'site.keywords':
			'Sabohat, Samantha, AI, SNN, spiking neural network, artificial intelligence, Uzbek language, neural network, continual learning, brain-inspired AI, machine learning',
		'site.locale': 'en_US',

		'nav.home': 'home',
		'nav.blog': 'posts',
		'nav.projects': 'projects',
		'nav.about': 'about',

		'home.intro.lead': "based, brain-inspired AI model. The goal — make the model learn on its own and significantly reduce the",
		'home.intro.cl': 'problem (catastrophic forgetting).',
		'home.info.architecture': 'architecture',
		'home.info.architectureValue': 'SNN · brain-inspired · spike-based',
		'home.info.goal': 'goal',
		'home.info.goalValue': 'self-learning + reduce CL forgetting',
		'home.info.experiments': 'experiments',
		'home.info.site': 'site',
		'home.divider.recent': 'recent posts',
		'home.allPosts': '→ all posts',

		'blog.heading': '/posts',
		'blog.subtitle': '// A dev journal on Sabohat — a brain-inspired SNN AI.',
		'blog.countSuffix': 'posts',
		'blog.back': '← posts',
		'blog.backAll': '← all posts',

		'about.heading': '/about',
		'about.title': 'About · Sabohat',
		'about.description':
			'Sabohat — a brain-inspired SNN (Spiking Neural Network) AI model. About the project goals, stages, and what gets written here.',

		'projects.heading': '/projects',
		'projects.subtitle': '// What I am currently working on.',
		'projects.title': 'Projects · Sabohat',
		'projects.description':
			'AI projects I am currently working on: Sabohat (brain-inspired SNN) and the main solvie.me domain.',
		'projects.statusActive': 'active',

		'404.title': 'Page not found · Sabohat',
		'404.description': 'The page you were looking for could not be found.',
		'404.text': 'The page you were looking for does not exist. It may have been renamed or removed.',
		'404.home': '← home',
		'404.art': ' ┌────────────────────────────┐\n │  404 · page not found      │\n │  ─ signal lost ─           │\n └────────────────────────────┘',

		'footer.copy': '© {year} sabohat',
		'footer.built': 'built with astro + ☕',
		'langSwitch.label': 'lang',
	},
} as const;

export type UIKey = keyof (typeof ui)[typeof defaultLang];

export function getLangFromUrl(url: URL): Lang {
	const [, seg] = url.pathname.split('/');
	if (seg in languages) return seg as Lang;
	return defaultLang;
}

export function useTranslations(lang: Lang) {
	return function t(key: UIKey): string {
		return (ui[lang] as Record<string, string>)[key] ?? (ui[defaultLang] as Record<string, string>)[key] ?? key;
	};
}

export function localizePath(path: string, lang: Lang): string {
	const clean = path.startsWith('/') ? path : `/${path}`;
	if (lang === defaultLang) return clean;
	return `/${lang}${clean === '/' ? '' : clean}`;
}

export function stripLangFromPath(pathname: string): string {
	const parts = pathname.split('/');
	if (parts[1] && parts[1] in languages) {
		parts.splice(1, 1);
	}
	const out = parts.join('/') || '/';
	return out;
}
