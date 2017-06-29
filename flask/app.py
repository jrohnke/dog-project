import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug import secure_filename
from dog_app import dog_breed_check

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

CHECKER = dog_breed_check()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    del_old_files()
    if request.method == 'POST':
        if 'image' in request.form:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        elif 'url' in request.form:
            url = request.form['url']
            try:
                filename = CHECKER.convert_url(url)
            except:
                return render_template('error.html')
        return redirect(url_for('uploaded_file', filename=filename))
    return render_template('uploader.html')


@app.route('/show/<filename>', methods=['GET', 'POST'])
def uploaded_file(filename):

    breed, isdog, gimage = CHECKER.check_breed(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    filename = '/uploads/' + filename
    if request.method == 'POST':
        if 'image' in request.form:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        elif 'url' in request.form:  
            url = request.form['url']
            try:
                filename = CHECKER.convert_url(url)
            except:
                return render_template('error.html')
        return redirect(url_for('uploaded_file', filename=filename))

    if isdog:
    	if len(breed)>1:
    		return render_template('template_dog_mix.html', filename=filename, breed1=breed[0], breed2=breed[1])
    	else:
    		return render_template('template_dog.html', filename=filename, breed=breed[0])
    elif gimage:
    	gimage = '/uploads/' + gimage
    	return render_template('template_human.html', filename=filename, breed=breed[0], gimage=gimage)
    else:
    	return render_template('template_nothing.html', filename=filename)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def del_old_files():
    from pathlib import Path
    import arrow
    import glob

    for fl in glob.glob("uploads/input*.jpg"):
        os.remove(fl)

    for fl in glob.glob("uploads/gsearch_*.jpg"):
        os.remove(fl)

    criticalTime = arrow.now().shift(minutes=-5)
    for item in Path(UPLOAD_FOLDER).glob('*'):
        if item.is_file():
            itemTime = arrow.get(item.stat().st_mtime)
            if itemTime < criticalTime:
                #remove it
                os.remove(str(item))

if __name__ == '__main__':
    app.run(host= '0.0.0.0')
    # app.run(debug=True)