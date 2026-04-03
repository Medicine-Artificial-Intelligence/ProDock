
document.addEventListener("DOMContentLoaded", function () {
  const end = document.querySelector(".navbar-header-items__end") || document.querySelector(".navbar-end-items") || document.querySelector(".bd-header .navbar");
  if (end && !document.getElementById("pd-fullscreen-toggle")) {
    const btn = document.createElement("button");
    btn.id = "pd-fullscreen-toggle";
    btn.className = "btn btn-sm";
    btn.setAttribute("title", "Toggle fullscreen");
    btn.setAttribute("aria-label", "Toggle fullscreen");
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H3v5"/><path d="M16 3h5v5"/><path d="M3 16v5h5"/><path d="M21 16v5h-5"/></svg>';
    btn.addEventListener("click", function () {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
    });
    const searchButton = end.querySelector('.search-button__button, .search-button-field, .search-button');
    if (searchButton && searchButton.parentElement) {
      searchButton.parentElement.before(btn);
    } else {
      end.appendChild(btn);
    }
  }
});
