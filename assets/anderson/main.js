const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const escapeHtml = str =>
    String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');

const formatDate = input => {
    const d = new Date(input);
    return isNaN(d) ? '' : `${MONTH_ABBR[d.getMonth()]} ${d.getDate()} ${d.getFullYear()}`;
};

const fetchLines = async url => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    const text = await res.text();
    return text.split('\n').map(l => l.trim()).filter(Boolean);
};

function renderPosts(posts, searchText = '') {
    const list = document.getElementById('post-list');
    if (!list) return;

    const q = searchText.toLowerCase();
    const filtered = q
        ? posts.filter(p =>
            String(p.title ?? '').toLowerCase().includes(q) ||
            (p.categories ?? []).some(c => c.toLowerCase().includes(q)) ||
            (p.tags ?? []).some(t => String(t).toLowerCase().includes(q))
          )
        : posts;

    list.innerHTML = filtered.map(({ date, categories = [], url = '#', title = '', tags = [] }) => {
        const safeUrl = escapeHtml(url);
        return `<li>
            <span class="post-date">${formatDate(date)}</span> -
            <span class="post-category">${escapeHtml(categories.join(' '))}</span>
            <a class="post-comment-count" href="${safeUrl}#disqus_thread"></a>
            <div>
                <a class="post-link post-tag" href="${safeUrl}" title="${escapeHtml(tags.join(', '))}">
                    ${escapeHtml(title)}
                </a>
            </div>
        </li>`;
    }).join('');

    if (typeof bootstrap !== 'undefined') {
        list.querySelectorAll('[title]').forEach(el => new bootstrap.Tooltip(el));
    }
}

function initPostSearch(posts) {
    renderPosts(posts);
    document.getElementById('search-input')?.addEventListener('input', function () {
        renderPosts(posts, this.value);
    });
}

async function loadBibleVerse() {
    try {
        const lines = await fetchLines('/assets/anderson/bible.csv');
        const el = document.getElementById('bible-statement');
        if (el && lines.length) {
            el.textContent = lines[Math.floor(Math.random() * lines.length)];
        }
    } catch (err) {
        console.warn('Bible verse load failed:', err.message);
    }
}

async function loadFastCategories(posts) {
    try {
        const cats = await fetchLines('/assets/anderson/fast_categories.csv');
        const container = document.getElementById('fast-categories');
        if (!container || !cats.length) return;

        container.innerHTML = cats
            .map(cat => `<button type="button" class="fast_category" data-cat="${escapeHtml(cat)}">${escapeHtml(cat)}</button>`)
            .join('');

        container.querySelectorAll('.fast_category').forEach(btn => {
            btn.addEventListener('click', () => {
                const cat = btn.dataset.cat;
                const input = document.getElementById('search-input');
                if (input) input.value = cat;
                renderPosts(posts, cat);
            });
        });
    } catch (err) {
        console.warn('Fast categories load failed:', err.message);
    }
}
