'use strict';

var gulp = require('gulp');
var csso = require('gulp-csso');
var uglify = require('gulp-uglify');
var concat = require('gulp-concat');
var sass = require('gulp-sass')(require('sass'));
var plumber = require('gulp-plumber');
var cp = require('child_process');
var imagemin = require('gulp-imagemin');
var browserSync = require('browser-sync').create();

function runSpawn(cmd, args, options, cb) {
  var finished = false;
  function finish(err) {
    if (!finished) {
      finished = true;
      cb(err);
    }
  }
  var child = cp.spawn(cmd, args, options);
  child.on('error', function () {
    finish(new Error('spawn failed: ' + cmd));
  });
  child.on('close', function (code) {
    finish(code === 0 ? null : new Error(cmd + ' exited ' + code));
  });
}

function jekyllBuild(cb) {
  var isWin = /^win/.test(process.platform);
  var cmd = isWin ? 'bundle.bat' : 'bundle';
  runSpawn(
    cmd,
    ['exec', 'jekyll', 'build'],
    { stdio: 'inherit', shell: isWin },
    function (bundleErr) {
      if (!bundleErr) {
        return cb();
      }
      if (isWin) {
        return cb(bundleErr);
      }
      runSpawn('jekyll', ['build'], { stdio: 'inherit' }, function (jErr) {
        cb(jErr || undefined);
      });
    }
  );
}

function jekyllBuildOptional(done) {
  jekyllBuild(function (err) {
    if (err) {
      console.warn('[gulp] Jekyll build skipped:', err.message);
      console.warn('[gulp] Run `bundle install` then `bundle exec jekyll build` to generate _site.');
    }
    done();
  });
}

function jekyllBuildTask(done) {
  jekyllBuild(done);
}

function reloadBrowser(done) {
  browserSync.reload();
  done();
}

function sassTask() {
  return gulp
    .src('src/styles/main.scss')
    .pipe(plumber())
    .pipe(sass().on('error', sass.logError))
    .pipe(csso())
    .pipe(gulp.dest('assets/css/'));
}

function fonts() {
  return gulp
    .src('src/fonts/**/*.{ttf,woff,woff2}')
    .pipe(plumber())
    .pipe(gulp.dest('assets/fonts/'));
}

function imageminTask() {
  return gulp
    .src('src/img/**/*.{jpg,png,gif}')
    .pipe(plumber())
    .pipe(
      imagemin({
        optimizationLevel: 3,
        progressive: true,
        interlaced: true,
      })
    )
    .pipe(gulp.dest('assets/img/'));
}

function js() {
  return gulp
    .src('src/js/**/*.js')
    .pipe(plumber())
    .pipe(concat('main.js'))
    .pipe(uglify())
    .pipe(gulp.dest('assets/js/'));
}

var jekyllReload = gulp.series(jekyllBuildTask, reloadBrowser);

function serve() {
  browserSync.init({
    server: { baseDir: '_site' },
  });
  gulp.watch('src/styles/**/*.scss', gulp.series(sassTask, jekyllReload));
  gulp.watch('src/js/**/*.js', gulp.series(js, reloadBrowser));
  gulp.watch('src/fonts/**/*.{ttf,woff,woff2}', gulp.series(fonts, reloadBrowser));
  gulp.watch('src/img/**/*.{jpg,png,gif}', gulp.series(imageminTask, jekyllReload));
  gulp.watch(
    ['*.html', '_includes/*.html', '_includes/*.md', '_layouts/*.html', '*.md', '_config.yml'],
    jekyllReload
  );
  return new Promise(function () {});
}

var compileAssets = gulp.parallel(js, sassTask, fonts);

exports.sass = sassTask;
exports.js = js;
exports.fonts = fonts;
exports.imagemin = imageminTask;
exports.jekyllBuild = jekyllBuildTask;
exports.build = gulp.series(compileAssets, imageminTask, jekyllBuildOptional);
exports.default = gulp.series(compileAssets, jekyllBuildTask, serve);
