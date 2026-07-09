(function () {
  const current = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".main-nav a").forEach((link) => {
    const href = link.getAttribute("href");
    if (href === current || (current === "" && href === "index.html")) {
      link.setAttribute("aria-current", "page");
    }
  });

  document.querySelectorAll("[data-check-id]").forEach((item) => {
    const id = "minimind-o-tutorial:" + item.dataset.checkId;
    item.checked = localStorage.getItem(id) === "1";
    item.addEventListener("change", () => {
      localStorage.setItem(id, item.checked ? "1" : "0");
    });
  });

  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "aria-hidden": "true" } });
  }
})();
