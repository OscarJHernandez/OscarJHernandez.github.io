/* sweetScroll for in-page anchor links */
document.addEventListener('DOMContentLoaded', function () {
  new SweetScroll({});
}, false);

document.addEventListener(
  'click',
  function (e) {
    var btn = e.target.closest('.js-email-reveal');
    if (!btn) return;
    var b64 = btn.getAttribute('data-email-b64');
    if (!b64) return;
    e.preventDefault();
    var wrap = btn.closest('.email-reveal-wrap');
    if (!wrap) return;
    try {
      var addr = atob(b64);
      var a = document.createElement('a');
      a.href = 'mailto:' + addr;
      a.textContent = addr;
      a.className = 'email-reveal-link';
      wrap.replaceChild(a, btn);
    } catch (err) {
      /* ignore invalid base64 */
    }
  },
  false
);

/* Blog sidebar section toggle */
document.addEventListener(
  'click',
  function (e) {
    var toggle = e.target.closest('.sidebar-section-toggle');
    if (!toggle) return;
    var expanded = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', expanded ? 'false' : 'true');
    var listId = toggle.getAttribute('aria-controls');
    var list = listId ? document.getElementById(listId) : null;
    if (list) {
      list.classList.toggle('is-collapsed', expanded);
    }
  },
  false
);
