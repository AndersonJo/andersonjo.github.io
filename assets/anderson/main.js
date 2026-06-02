const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function andersonDate(input) {
    const date = new Date(input);
    return MONTH_ABBR[date.getMonth()] + ' ' + date.getDate() + ' ' + date.getFullYear();
}

function randint(max, min = 0) {
    return Math.floor(Math.random() * (max - min + 1) + min);
}

async function fetchCSVLines(url) {
    const response = await fetch(url);
    const text = await response.text();
    return text.split('\n').filter(line => line.trim().length > 0);
}

function renderPosts(posts, searchText) {
    const list = document.getElementById('post-list');
    if (!list) return;

    const query = (searchText || '').toLowerCase();
    const filtered = query
        ? posts.filter(p =>
            p.title.toLowerCase().includes(query) ||
            p.categories.some(c => c.toLowerCase().includes(query)) ||
            (p.tags && p.tags.some(t => t.toLowerCase().includes(query)))
          )
        : posts;

    list.innerHTML = filtered.map(post => {
        const tagTitle = post.tags ? post.tags.join(', ') : '';
        return `<li>
            <span class="post-date">${andersonDate(post.date)}</span> -
            <span class="post-category">${post.categories.join(' ')}</span>
            <a class="post-comment-count" href="${post.url}#disqus_thread"></a>
            <div>
                <a class="post-link post-tag" href="${post.url}"
                   data-bs-toggle="tooltip" data-bs-placement="right"
                   title="${tagTitle}">
                    ${post.title}
                </a>
            </div>
        </li>`;
    }).join('');

    list.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
        new bootstrap.Tooltip(el);
    });
}

document.addEventListener('DOMContentLoaded', async function () {
    const posts = typeof global_post_data !== 'undefined' ? global_post_data : [];
    const searchInput = document.getElementById('search-input');

    if (searchInput) {
        renderPosts(posts, '');
        searchInput.addEventListener('input', function () {
            renderPosts(posts, this.value);
        });
    }

    try {
        const statements = await fetchCSVLines('/assets/anderson/bible.csv');
        const bibleEl = document.getElementById('bible-statement');
        if (bibleEl && statements.length > 0) {
            bibleEl.textContent = statements[randint(statements.length - 1)];
        }
    } catch (_) {}

    try {
        const categories = await fetchCSVLines('/assets/anderson/fast_categories.csv');
        const catContainer = document.getElementById('fast-categories');
        if (catContainer && categories.length > 0) {
            catContainer.innerHTML = categories.map(cat =>
                `<button type="button" class="fast_category" data-category="${cat}">${cat}</button>`
            ).join('');
            catContainer.querySelectorAll('.fast_category').forEach(btn => {
                btn.addEventListener('click', function () {
                    const cat = this.dataset.category;
                    if (searchInput) {
                        searchInput.value = cat;
                        renderPosts(posts, cat);
                    }
                });
            });
        }
    } catch (_) {}
});
