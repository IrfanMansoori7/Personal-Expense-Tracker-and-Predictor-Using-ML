# backend/app.py
import os
import sys
from flask import Flask, render_template, request, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import pandas as pd
import io
import warnings
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from dateutil import parser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from finance_advisor import FinanceAdvisor, CATEGORIES  # Changed import here

app = Flask(__name__,
          template_folder=os.path.join('..', 'frontend', 'templates'),
          static_folder=os.path.join('..', 'frontend', 'static'))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key_here'
db = SQLAlchemy(app)

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.Integer, nullable=False)
    date = db.Column(db.String(10), nullable=False)
    description = db.Column(db.String(100), default='')

warnings.filterwarnings('ignore')

def generate_graphs(transactions):
    df = pd.DataFrame([{
        'amount': t.amount,
        'category': t.category,
        'date': pd.to_datetime(t.date, errors='coerce'), # Handle potential invalid dates
        'description': t.description
    } for t in transactions])
    df.dropna(subset=['date'], inplace=True) # Remove rows with invalid dates

    graphs = []
    titles = []

    if df.empty:
        return graphs, titles

    # Graph 1: Monthly Spending Trend
    plt.figure(figsize=(12, 6))
    monthly_spend = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum()
    ax = monthly_spend.plot(kind='line', title='Monthly Spending Trend', color='#3498db')
    plt.xlabel("Month")
    plt.ylabel("Total Spending")
    plt.grid(True)
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    graphs.append(base64.b64encode(img.getvalue()).decode('utf-8'))
    titles.append("Monthly Spending Trend")
    plt.close()

    # Graph 2: Category Distribution
    plt.figure(figsize=(12, 6))
    df['category_name'] = df['category'].map(CATEGORIES)
    df.groupby('category_name')['amount'].sum().plot(
        kind='pie',
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    plt.title('Spending by Category')
    plt.ylabel('') # Hide default y-label for pie chart
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    graphs.append(base64.b64encode(img.getvalue()).decode('utf-8'))
    titles.append("Spending by Category")
    plt.close()

    # Graph 3: Weekly Pattern (Bar Chart)
    plt.figure(figsize=(12, 6))
    df['day_of_week'] = df['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
    df.groupby('day_of_week')['amount'].sum().plot(
        kind='bar',
        color='#2ecc71'
    )
    plt.title('Weekly Spending Pattern')
    plt.xlabel("Day of the Week")
    plt.ylabel("Total Spending")
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    graphs.append(base64.b64encode(img.getvalue()).decode('utf-8'))
    titles.append("Weekly Spending Pattern")
    plt.close()

    # Graph 4: Monthly Trend (Bar Chart)
    plt.figure(figsize=(12, 6))
    df['month'] = df['date'].dt.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
    df.groupby('month')['amount'].sum().plot(
        kind='bar',
        color='#9b59b6'
    )
    plt.title('Monthly Spending Trend')
    plt.xlabel("Month")
    plt.ylabel("Total Spending")
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    graphs.append(base64.b64encode(img.getvalue()).decode('utf-8'))
    titles.append("Monthly Spending Trend")
    plt.close()

    return graphs, titles

@app.route('/')
def home():
    all_transactions = Transaction.query.order_by(Transaction.date.desc()).all()
    # Filter out transactions with potentially invalid dates before passing to FinanceAdvisor
    valid_transactions = [t for t in all_transactions if is_valid_date(t.date)]
    advisor = FinanceAdvisor(valid_transactions)
    advice = advisor.generate_insights()
    recent_transactions = valid_transactions[:10] # Use filtered transactions for recent list

    return render_template('index.html',
                           transactions=recent_transactions,
                           all_transactions_count=len(valid_transactions),
                           advice=advice,
                           categories=CATEGORIES,
                           today=date.today().isoformat())

@app.route('/all-transactions')
def all_transactions():
    transactions = Transaction.query.order_by(Transaction.date.desc()).all()
    # Filter out transactions with potentially invalid dates
    valid_transactions = [t for t in transactions if is_valid_date(t.date)]
    return render_template('all_transactions.html',
                           transactions=valid_transactions,
                           categories=CATEGORIES)

@app.route('/delete-all', methods=['POST'])
def delete_all_transactions():
    try:
        num_rows_deleted = db.session.query(Transaction).delete()
        db.session.commit()
        flash(f"Successfully deleted {num_rows_deleted} transactions!")
    except Exception as e:
        db.session.rollback()
        flash("Error deleting transactions")
    return redirect(url_for('home')) # Redirect to home after deleting all

@app.route('/graphs')
def show_graphs():
    transactions = Transaction.query.order_by(Transaction.date.desc()).all()
    # Filter out transactions with potentially invalid dates for graph generation
    valid_transactions = [t for t in transactions if is_valid_date(t.date)]
    graphs, titles = generate_graphs(valid_transactions)
    return render_template('graphs.html', graphs=graphs, titles=titles)

def is_valid_date(date_string):
    try:
        parser.parse(date_string)
        return True
    except (ValueError, TypeError):
        return False

@app.route('/add', methods=['POST'])
def add_transaction():
    try:
        amount = float(request.form['amount'])
        category = int(request.form['category'])
        transaction_date_str = request.form['date']
        description = request.form.get('description', '')

        if amount <= 0:
            flash("Amount must be positive")
            return redirect('/')
        if category not in CATEGORIES:
            flash("Select a valid category (1-8)")
            return redirect('/')

        try:
            input_date = datetime.strptime(transaction_date_str, '%Y-%m-%d').date()
            if input_date > date.today():
                flash("Date cannot be in future")
                return redirect('/')
            transaction_date = input_date.isoformat() # Store in 'YYYY-MM-DD' format
        except ValueError:
            flash("Use YYYY-MM-DD date format")
            return redirect('/')

        new_trans = Transaction(
            amount=amount,
            category=category,
            date=transaction_date,
            description=description
        )
        db.session.add(new_trans)
        db.session.commit()
        flash("Transaction added successfully!")
        return redirect('/')

    except ValueError:
        flash("Invalid amount or category format")
        return redirect('/')
    except Exception as e:
        flash("Error adding transaction. Please try again.")
        return redirect('/')

@app.route('/upload', methods=['POST'])
def upload_csv():
    try:
        if 'csv_file' not in request.files:
            flash('No file selected')
            return redirect('/')

        file = request.files['csv_file']
        if file.filename == '':
            flash('No file selected')
            return redirect('/')

        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8')))

            if not all(col in df.columns for col in ['amount', 'category', 'date']):
                flash('CSV must contain: amount, category, date columns')
                return redirect('/')

            success_count = 0
            error_rows = []
            for index, row in df.iterrows():
                try:
                    amount = float(row['amount'])
                    category = int(row['category'])
                    date_str = str(row['date'])
                    try:
                        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        transaction_date = parsed_date.isoformat()
                    except ValueError:
                        try:
                            parsed_date = parser.parse(date_str).date()
                            transaction_date = parsed_date.isoformat()
                        except ValueError:
                            error_rows.append(f"Row {index + 2}: Invalid date format '{date_str}'")
                            continue

                    description = str(row.get('description', ''))

                    new_trans = Transaction(
                        amount=amount,
                        category=category,
                        date=transaction_date,
                        description=description
                    )
                    db.session.add(new_trans)
                    success_count += 1
                except (ValueError, TypeError) as e:
                    error_rows.append(f"Row {index + 2}: Invalid data - {e}")
                    continue

            db.session.commit()
            flash(f'Successfully added {success_count} transactions from CSV!')
            if error_rows:
                flash("<br>The following rows had errors and were skipped:<br>" + "<br>".join(error_rows))
            return redirect('/')
        else:
            flash('Please upload a CSV file')
            return redirect('/')
    except Exception as e:
        flash('Error processing CSV file')
        return redirect('/')

@app.route('/delete/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    try:
        transaction = Transaction.query.get_or_404(transaction_id)
        db.session.delete(transaction)
        db.session.commit()
        flash("Transaction deleted successfully!")
    except:
        flash("Error deleting transaction.")
    return redirect(request.referrer or '/')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)