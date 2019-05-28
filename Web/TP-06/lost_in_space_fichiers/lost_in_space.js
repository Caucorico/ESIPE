"use strict";

let game;

class Game {
	constructor( width, height ) {
		this.width = width;
		this.height = height;
		this.bullets = new Array();
		this.aliens = Array();
	}

	initPlayerShip() {
		this.playerShip = new Ship( this.width/2, this.height, 0, 0, 0, 0 );
	}

	initAliens() {
		for ( let i = 0 ; i < 300 ; i++ )
		{
			let x;
			let y;
			if ( Math.floor(Math.random()*2)%2 == 0 ) {
				x = 0;
			} else {
				x = 600;
			}
			y = Math.floor(Math.random()*601);

			console.log(x);
			console.log(y);

			let alien = new Ship( x, y, 5, 5, 0, 0);
			this.aliens.unshift(alien);
		}
	}

	treatBullet( bullet ) {
		bullet.applyAcceleration();
		bullet.applySpeed();
	}

	treatBullets() {
		this.bullets.map(this.treatBullet);
	}

	displayBullet( bullet ) {
		bullet.drawBullet();
	}

	displayBullets() {
		this.bullets.map(this.displayBullet);
	}

	displayAlien( alien ) {
		let canvas = document.getElementById("game_area");
		let ctx = canvas.getContext("2d");
		alien.drawShip(ctx);
	}

	displayAliens() {
		this.aliens.map(this.displayAlien);
	}
}

class Ship {
	constructor( x, y, speedX, speedY, accelerationX, accelerationY ) {
		this.x = x;
		this.y = y;
		this.speedX = 0;
		this.speedY = 0;
		this.accelerationX = 0;
		this.accelerationY = 0;
	}

	drawShip( ctx ) {
		ctx.strokeStyle = "#FF0000";
		ctx.beginPath();
		ctx.moveTo(this.x, this.y);
		ctx.lineTo(this.x+5, this.y+10);
		ctx.lineTo(this.x-5, this.y+10);
		ctx.lineTo(this.x, this.y);
		ctx.stroke();
	}

	applyAcceleration() {
		this.speedX = clamp(this.speedX + this.accelerationX, -2.5, 2.5);
		this.speedY = clamp(this.speedY + this.accelerationY, -2.5, 2.5);
	}

	applySpeed() {
		this.x += this.speedX;
		this.y += this.speedY;
	}
}

class Bullet {
	constructor( x, y, speedX, speedY, accelerationX, accelerationY ) {
		this.x = x;
		this.y = y;
		this.speedX = speedX;
		this.speedY = speedY;
		this.accelerationX = accelerationX;
		this.accelerationY = accelerationY;
	}

	applyAcceleration() {
		this.speedX += this.accelerationX;
		this.speedY += this.accelerationY;
	}

	applySpeed() {
		this.x += this.speedX;
		this.y += this.speedY;
	}

	drawBullet() {
		/* TODO : degager ce canvas */
		let canvas = document.getElementById("game_area");
		let ctx = canvas.getContext("2d");
		ctx.beginPath();
		ctx.arc(this.x, this.y, 2, 0 , 2 * Math.PI);
		ctx.fillStyle = "white";
		ctx.fill();
	}
}

function clamp( a, min, max )
{
	if ( a > max ) return max;
	else if ( a < min ) return min;
	else return a;
}

window.onload = function() {
	let canvas = document.getElementById("game_area");
	let context = canvas.getContext("2d");
	context.fillStyle = "black";
	game = new Game( 600, 600 );
	game.initPlayerShip();
	game.initAliens();

	let map = {}; // You could also use an array
	onkeydown = onkeyup = function(e){
	    e = e || event; // to deal with IE
	    map[e.keyCode] = e.type == 'keydown';

	    if ( map[38] && map[40] ) /* up + down*/
	    {
	    	game.playerShip.accelerationY = 0;
	    }

	    if ( !map[38] && !map[40] ) /* ~up + ~down*/
	    {
	    	game.playerShip.accelerationY = 0;
	    }

	    if ( map[37] && map[39] ) /* left + right*/
	    {
	    	game.playerShip.accelerationX = 0;
	    }

	    if ( !map[37] && !map[39] ) /* ~left + ~right*/
	    {
	    	game.playerShip.accelerationX = 0;
	    }
	    
	    if ( map[38] ) {/* up */
	    	game.playerShip.accelerationY = clamp( game.playerShip.accelerationY-0.3, -1, 1);
	    }
	    if ( map[40] ) {/* down */
	    	game.playerShip.accelerationY = clamp( game.playerShip.accelerationY+0.3, -1, 1);;
	    }

	    if ( map[37] ) {/* left */
	    	game.playerShip.accelerationX = clamp( game.playerShip.accelerationX-0.3, -1, 1);;
	    }
	    if ( map[39] ) {/* right */
	    	game.playerShip.accelerationX = clamp( game.playerShip.accelerationX+0.3, -1, 1);;
	    }

	    if ( map[32] ) { /* space */
	    	let newBullet = new Bullet( game.playerShip.x, game.playerShip.y, 0, clamp(game.playerShip.speedY, game.playerShip.speedY, -0.5), 0, 0 );
	    	game.bullets.unshift(newBullet);
	    	console.log(game.bullets);
	    }
	}

	context.save();
	game.playerShip.drawShip( context );

	this.interval = setInterval(() => {
		context.clearRect(0,0,600,600);
		game.playerShip.applyAcceleration();
		game.playerShip.applySpeed();
		game.playerShip.drawShip( context );
		game.treatBullets();
		game.displayBullets();
		game.displayAliens();
	}, 20);
};
