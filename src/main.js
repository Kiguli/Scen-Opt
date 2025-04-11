document.getElementById('lp-tab').addEventListener('shown.bs.tab', function () {
    saveFormValues('qp');
    saveFormValues('sdp');
    loadFormValues();
    updateLatexText('lp');
});
document.getElementById('qp-tab').addEventListener('shown.bs.tab', function () {
    saveFormValues('lp');
    saveFormValues('sdp');
    loadFormValues();
    updateLatexText('qp');
});
document.getElementById('sdp-tab').addEventListener('shown.bs.tab', function () {
    saveFormValues('lp');
    saveFormValues('qp');
    loadFormValues();
    updateLatexText('sdp');
});

document.addEventListener('DOMContentLoaded', function () {
    //TODO: changing tabs doesn't store values

    /* Add event listener for form submission */
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function (event) {
            event.preventDefault(); // Prevent the default form submission

            const activeTab = document.querySelector('.nav-tabs .active').getAttribute('href').substring(1);
            const formData = new FormData(this); // Use the current form
            formData.append('active_tab', activeTab); // Include the active tab in the form data

            fetch('/solve', {
                method: 'POST',
                body: formData
            })
                .then(response => response.text())
                .then(html => {
                    // Update the result box dynamically
                    document.getElementById('result-box').innerHTML = html;
                })
                .catch(error => console.error('Error:', error));
        });
    });
});