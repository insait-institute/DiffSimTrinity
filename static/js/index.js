document.addEventListener('DOMContentLoaded', () => {
  const burgers = document.querySelectorAll('.navbar-burger');

  burgers.forEach((burger) => {
    const navbar = burger.closest('.navbar');
    const menu = navbar ? navbar.querySelector('.navbar-menu') : null;

    if (!menu) {
      return;
    }

    burger.addEventListener('click', () => {
      burger.classList.toggle('is-active');
      menu.classList.toggle('is-active');
    });
  });
});
